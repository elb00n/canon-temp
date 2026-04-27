"""Centralized SQLite helper for the Canon inspection dashboard.

DB 파일 위치: <backend_root>/data/db/factory_test.db

테이블:
    inspection_log     — 카메라/이미지/override 검사 이력
    sequence_run       — 영상 순차 판정 실행 요약
    sequence_run_video — 각 영상별 결과
    sequence_run_target — 각 영상 내 target 단계별 결과

이 모듈은 app 패키지에 의존하지 않아 독립적으로 사용 가능합니다.
api_router.py 와 service layer 는 이 모듈을 import 하여 사용하세요.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

# ── 경로 해결 ──────────────────────────────────────────────────────────────────
# backend/db/database.py  →  parents[1] = backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_DB_DIR = _BACKEND_ROOT / "data" / "db"
DB_PATH = _DB_DIR / "factory_test.db"
DEFAULT_DB_PATH = DB_PATH

# ── 스키마 ─────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS inspection_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    source_type     TEXT    NOT NULL DEFAULT 'camera',
    confirmed_state TEXT    NOT NULL DEFAULT 'Unknown',
    predicted_label TEXT    NOT NULL DEFAULT '',
    confidence      REAL    NOT NULL DEFAULT 0.0,
    anomaly_flag    INTEGER NOT NULL DEFAULT 0,
    file_path       TEXT    NOT NULL DEFAULT '',
    cam_id          TEXT    NOT NULL DEFAULT '',
    target_idx      INTEGER NOT NULL DEFAULT 0,
    extra_json      TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_inspection_log_timestamp
    ON inspection_log (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_inspection_log_cam_id
    ON inspection_log (cam_id);

CREATE TABLE IF NOT EXISTS sequence_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    run_root TEXT,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequence_run_video (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    video_name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    completed INTEGER NOT NULL,
    processed_frames INTEGER NOT NULL,
    confirmed_targets INTEGER NOT NULL,
    total_detections INTEGER NOT NULL,
    video_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES sequence_run(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sequence_run_target (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    target_name TEXT NOT NULL,
    completed INTEGER NOT NULL,
    start_frame INTEGER,
    confirmed_frame INTEGER,
    processed_frames INTEGER NOT NULL,
    detections_seen INTEGER NOT NULL,
    target_json TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES sequence_run_video(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sequence_run_timestamp
    ON sequence_run (run_timestamp);

CREATE INDEX IF NOT EXISTS idx_sequence_run_video_run_id
    ON sequence_run_video (run_id);

CREATE INDEX IF NOT EXISTS idx_sequence_run_target_video_id
    ON sequence_run_target (video_id);
"""


# ── 연결 헬퍼 ─────────────────────────────────────────────────────────────────
def _ensure_db_dir() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """새 SQLite 연결을 반환합니다. Row 팩토리와 외래키가 활성화됩니다."""
    _ensure_db_dir()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """sequence_db.py 호환용 연결 헬퍼입니다."""
    return get_connection(db_path)


@contextmanager
def db_conn(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """컨텍스트 매니저 방식으로 커넥션을 사용하고 자동 커밋/롤백합니다."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── 초기화 ────────────────────────────────────────────────────────────────────
def initialize(db_path: Path = DB_PATH) -> None:
    """DB 파일과 테이블을 생성합니다. 서버 시작 시 1회 호출하세요."""
    with db_conn(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)


def initialize_sequence_db(db_path: Path = DB_PATH) -> None:
    """sequence DB 호환용 초기화 헬퍼입니다."""
    initialize(db_path)


# ── Row → dict 변환 ───────────────────────────────────────────────────────────
def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # anomaly_flag 를 bool 로 변환
    if "anomaly_flag" in d:
        d["anomaly_flag"] = bool(d["anomaly_flag"])
    return d


# ── CRUD ──────────────────────────────────────────────────────────────────────
def insert_log(
    *,
    source_type: str = "camera",
    confirmed_state: str = "Unknown",
    predicted_label: str = "",
    confidence: float = 0.0,
    anomaly_flag: bool = False,
    file_path: str = "",
    cam_id: str = "",
    target_idx: int = 0,
    extra: dict[str, Any] | None = None,
    timestamp: str | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """inspection_log 에 새 행을 삽입하고 row id 를 반환합니다."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    extra_json = json.dumps(extra or {}, ensure_ascii=False)
    with db_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO inspection_log
                (timestamp, source_type, confirmed_state, predicted_label,
                 confidence, anomaly_flag, file_path, cam_id, target_idx, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                source_type,
                confirmed_state,
                predicted_label,
                confidence,
                1 if anomaly_flag else 0,
                file_path,
                cam_id,
                target_idx,
                extra_json,
            ),
        )
        return int(cur.lastrowid)  # type: ignore[arg-type]


def get_logs(
    offset: int = 0,
    limit: int = 30,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """최신순으로 정렬된 inspection_log 목록을 반환합니다."""
    with db_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM inspection_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def get_log_by_id(log_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """id 로 단일 행을 조회합니다. 없으면 None 을 반환합니다."""
    with db_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM inspection_log WHERE id = ?", (log_id,)
        ).fetchone()
    return row_to_dict(row) if row else None


def update_log(
    log_id: int,
    *,
    confirmed_state: str | None = None,
    predicted_label: str | None = None,
    confidence: float | None = None,
    anomaly_flag: bool | None = None,
    timestamp: str | None = None,
    db_path: Path = DB_PATH,
) -> bool:
    """log_id 에 해당하는 행을 부분 업데이트합니다. 성공 시 True 반환."""
    fields: list[str] = []
    values: list[Any] = []

    if confirmed_state is not None:
        fields.append("confirmed_state = ?")
        values.append(confirmed_state)
    if predicted_label is not None:
        fields.append("predicted_label = ?")
        values.append(predicted_label)
    if confidence is not None:
        fields.append("confidence = ?")
        values.append(confidence)
    if anomaly_flag is not None:
        fields.append("anomaly_flag = ?")
        values.append(1 if anomaly_flag else 0)

    # 항상 timestamp 갱신
    fields.append("timestamp = ?")
    values.append(timestamp or datetime.now(timezone.utc).isoformat())

    if not fields:
        return False

    values.append(log_id)
    sql = f"UPDATE inspection_log SET {', '.join(fields)} WHERE id = ?"  # noqa: S608

    with db_conn(db_path) as conn:
        cur = conn.execute(sql, values)
    return cur.rowcount > 0


def delete_log(log_id: int, db_path: Path = DB_PATH) -> bool:
    """log_id 에 해당하는 행을 삭제합니다. 성공 시 True 반환."""
    with db_conn(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM inspection_log WHERE id = ?", (log_id,)
        )
    return cur.rowcount > 0


def count_logs(db_path: Path = DB_PATH) -> int:
    """전체 inspection_log 행 수를 반환합니다."""
    with db_conn(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM inspection_log").fetchone()
    return int(row["cnt"])


@dataclass(slots=True)
class StoredSequenceRun:
    id: int
    run_timestamp: str
    run_root: str
    summary_json: str


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_sequence_run_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary")
    if isinstance(summary, dict):
        return summary
    return {
        key: value
        for key, value in result.items()
        if key not in {"videos", "run_root", "run_timestamp"}
    }


def _sequence_run_row_to_dict(
    row: sqlite3.Row,
    *,
    video_id: int,
    target_total: int,
) -> dict[str, Any]:
    data = dict(row)
    completed = bool(data.get("completed"))
    confirmed_targets = int(data.get("confirmed_targets") or 0)
    video_json = data.get("video_json") or "{}"

    predicted_label = "unknown"
    confidence = 0.0
    try:
        video_data = json.loads(video_json)
        targets = video_data.get("targets", [])
        completed_targets = [target for target in targets if target.get("completed")]
        ref = completed_targets[-1] if completed_targets else (targets[-1] if targets else {})
        predicted_label = str(ref.get("target_name", "unknown"))
        confidence = float(ref.get("last_score", 0.0))
    except Exception:
        pass

    return {
        "id": video_id + 100_000,
        "timestamp": data["timestamp"],
        "source_type": "sequence_run",
        "confirmed_state": f"Complete_{confirmed_targets}of{target_total}" if completed else f"Partial_{confirmed_targets}of{target_total}",
        "predicted_label": predicted_label,
        "confidence": confidence,
        "anomaly_flag": not completed,
        "file_path": data.get("video_path", ""),
        "cam_id": "",
        "target_idx": confirmed_targets,
        "_seq_video_id": video_id,
        "video_name": data.get("video_name", ""),
        "output_dir": data.get("output_dir", ""),
        "completed": completed,
        "processed_frames": int(data.get("processed_frames") or 0),
        "confirmed_targets": confirmed_targets,
        "total_detections": int(data.get("total_detections") or 0),
        "video_json": video_json,
    }


def upsert_sequence_run(result: dict[str, Any], db_path: Path = DB_PATH) -> int:
    """Sequence run 결과를 저장하고 run id 를 반환합니다."""
    initialize(db_path)

    summary = _extract_sequence_run_summary(result)
    run_timestamp = str(
        result.get("run_timestamp")
        or summary.get("run_timestamp")
        or datetime.now(timezone.utc).isoformat()
    )
    run_root = str(result.get("run_root") or "")
    created_at = datetime.now(timezone.utc).isoformat()
    summary_json = _json_dumps(summary)

    with db_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sequence_run (run_timestamp, created_at, run_root, summary_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_timestamp) DO UPDATE SET
                created_at = excluded.created_at,
                run_root = excluded.run_root,
                summary_json = excluded.summary_json
            """,
            (run_timestamp, created_at, run_root, summary_json),
        )
        run_row = conn.execute(
            "SELECT id FROM sequence_run WHERE run_timestamp = ?",
            (run_timestamp,),
        ).fetchone()
        if run_row is None:
            raise RuntimeError("failed to resolve sequence_run id after upsert")

        run_id = int(run_row["id"])
        conn.execute("DELETE FROM sequence_run_video WHERE run_id = ?", (run_id,))

        videos = result.get("videos") or []
        for video in videos:
            if not isinstance(video, dict):
                continue
            cursor = conn.execute(
                """
                INSERT INTO sequence_run_video (
                    run_id, video_name, video_path, output_dir, completed,
                    processed_frames, confirmed_targets, total_detections, video_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(video.get("video_name", "")),
                    str(video.get("video_path", "")),
                    str(video.get("output_dir", "")),
                    1 if bool(video.get("completed")) else 0,
                    int(video.get("processed_frames", 0)),
                    int(video.get("confirmed_targets", 0)),
                    int(video.get("total_detections", 0)),
                    _json_dumps(video),
                ),
            )
            video_id = int(cursor.lastrowid)
            conn.execute("DELETE FROM sequence_run_target WHERE video_id = ?", (video_id,))

            targets = video.get("targets") or []
            for target_index, target in enumerate(targets):
                if not isinstance(target, dict):
                    continue
                conn.execute(
                    """
                    INSERT INTO sequence_run_target (
                        video_id, target_index, target_name, completed, start_frame, confirmed_frame,
                        processed_frames, detections_seen, target_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video_id,
                        target_index,
                        str(target.get("target_name", "")),
                        1 if bool(target.get("completed")) else 0,
                        target.get("start_frame"),
                        target.get("confirmed_frame"),
                        int(target.get("processed_frames", 0)),
                        int(target.get("detections_seen", 0)),
                        _json_dumps(target),
                    ),
                )

        return run_id


def get_sequence_runs(
    offset: int = 0,
    limit: int = 50,
    *,
    total_targets: int = 4,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """sequence_run_video 데이터를 inspection_log 형태로 반환합니다."""
    with db_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                srv.id,
                sr.created_at AS timestamp,
                srv.video_name,
                srv.video_path,
                srv.output_dir,
                srv.completed,
                srv.processed_frames,
                srv.confirmed_targets,
                srv.total_detections,
                srv.video_json,
                sr.summary_json,
                sr.run_root
            FROM sequence_run_video srv
            JOIN sequence_run sr ON srv.run_id = sr.id
            ORDER BY srv.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return [
        _sequence_run_row_to_dict(row, video_id=int(row["id"]), target_total=total_targets)
        for row in rows
    ]


__all__ = [
    "DB_PATH",
    "DEFAULT_DB_PATH",
    "StoredSequenceRun",
    "initialize",
    "initialize_sequence_db",
    "get_connection",
    "connect",
    "db_conn",
    "row_to_dict",
    "insert_log",
    "get_logs",
    "get_log_by_id",
    "update_log",
    "delete_log",
    "count_logs",
    "upsert_sequence_run",
    "get_sequence_runs",
]
