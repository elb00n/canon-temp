"""SQLite metadata logging for operational inference runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.paths import DB_DIR
from app.service.operational_types import ArtifactRecord, FrameInferenceResult


DEFAULT_OPERATIONAL_DB_PATH = DB_DIR / "operational_runs.sqlite3"


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
	id TEXT PRIMARY KEY,
	created_at TEXT NOT NULL,
	mode TEXT NOT NULL,
	scenario TEXT NOT NULL,
	metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_results (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id TEXT NOT NULL,
	frame_index INTEGER NOT NULL,
	timestamp TEXT NOT NULL,
	screen_detected INTEGER NOT NULL,
	bbox_json TEXT,
	detector_confidence REAL NOT NULL,
	target1_score REAL NOT NULL,
	target2_score REAL NOT NULL,
	target3_score REAL NOT NULL,
	target4_score REAL NOT NULL,
	thresholds_json TEXT NOT NULL,
	passed_targets_json TEXT NOT NULL,
	predicted_label TEXT NOT NULL,
	decision_type TEXT NOT NULL,
	ambiguous INTEGER NOT NULL,
	unknown INTEGER NOT NULL,
	reinspect_needed INTEGER NOT NULL,
	reinspect_performed INTEGER NOT NULL,
	effective_label TEXT NOT NULL,
	final_label TEXT NOT NULL,
	state_machine_allowed INTEGER NOT NULL,
	response_json TEXT NOT NULL,
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sequence_events (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id TEXT NOT NULL,
	frame_index INTEGER NOT NULL,
	timestamp TEXT NOT NULL,
	event_type TEXT NOT NULL,
	predicted_label TEXT NOT NULL,
	final_label TEXT NOT NULL,
	details_json TEXT NOT NULL,
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id TEXT NOT NULL,
	frame_index INTEGER NOT NULL,
	artifact_type TEXT NOT NULL,
	path TEXT NOT NULL,
	metadata_json TEXT NOT NULL,
	FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_frame_results_session ON frame_results(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_sequence_events_session ON sequence_events(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id, frame_index);
"""


def _json_dumps(payload: Any) -> str:
	return json.dumps(payload, ensure_ascii=False, indent=2)


def connect(db_path: Path = DEFAULT_OPERATIONAL_DB_PATH) -> sqlite3.Connection:
	db_path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(db_path)
	connection.row_factory = sqlite3.Row
	connection.execute("PRAGMA foreign_keys = ON;")
	return connection


def initialize(db_path: Path = DEFAULT_OPERATIONAL_DB_PATH) -> None:
	with connect(db_path) as connection:
		connection.executescript(SCHEMA_SQL)


class OperationalLogStore:
	def __init__(self, db_path: Path = DEFAULT_OPERATIONAL_DB_PATH) -> None:
		self.db_path = db_path
		initialize(self.db_path)

	def upsert_session(
		self,
		session_id: str,
		*,
		mode: str,
		scenario: str,
		metadata: dict[str, Any] | None = None,
	) -> None:
		with connect(self.db_path) as connection:
			connection.execute(
				"""
				INSERT INTO sessions (id, created_at, mode, scenario, metadata_json)
				VALUES (?, ?, ?, ?, ?)
				ON CONFLICT(id) DO UPDATE SET
					mode = excluded.mode,
					scenario = excluded.scenario,
					metadata_json = excluded.metadata_json
				""",
				(
					session_id,
					datetime.now(timezone.utc).isoformat(),
					mode,
					scenario,
					_json_dumps(metadata or {}),
				),
			)
			connection.commit()

	def insert_frame_result(self, result: FrameInferenceResult) -> int:
		response = result.response_dict()
		bbox_json = None if result.detection.bbox is None else _json_dumps(list(result.detection.bbox))
		with connect(self.db_path) as connection:
			cursor = connection.execute(
				"""
				INSERT INTO frame_results (
					session_id, frame_index, timestamp, screen_detected, bbox_json, detector_confidence,
					target1_score, target2_score, target3_score, target4_score, thresholds_json,
					passed_targets_json, predicted_label, decision_type, ambiguous, unknown,
					reinspect_needed, reinspect_performed, effective_label, final_label,
					state_machine_allowed, response_json
				)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				(
					result.session_id,
					result.frame_index,
					result.timestamp,
					1 if result.detection.screen_detected else 0,
					bbox_json,
					result.detection.detector_confidence,
					float(result.scores.get("Target1", 0.0)),
					float(result.scores.get("Target2", 0.0)),
					float(result.scores.get("Target3", 0.0)),
					float(result.scores.get("Target4", 0.0)),
					_json_dumps(result.thresholds),
					_json_dumps(result.decision.passed_targets),
					result.decision.predicted_label,
					result.decision.decision_type,
					1 if result.decision.ambiguous else 0,
					1 if result.decision.unknown else 0,
					1 if result.decision.reinspect_needed else 0,
					1 if result.reinspect_performed else 0,
					result.state.effective_label,
					result.state.final_label,
					1 if result.state.state_machine_allowed else 0,
					_json_dumps(response),
				),
			)
			connection.execute(
				"""
				INSERT INTO sequence_events (
					session_id, frame_index, timestamp, event_type, predicted_label, final_label, details_json
				)
				VALUES (?, ?, ?, ?, ?, ?, ?)
				""",
				(
					result.session_id,
					result.frame_index,
					result.timestamp,
					result.state.event_type,
					result.decision.predicted_label,
					result.state.final_label,
					_json_dumps(result.state.as_dict()),
				),
			)
			for artifact in result.artifacts:
				self._insert_artifact(connection, result.session_id, result.frame_index, artifact)
			connection.commit()
			return int(cursor.lastrowid)

	def _insert_artifact(
		self,
		connection: sqlite3.Connection,
		session_id: str,
		frame_index: int,
		artifact: ArtifactRecord,
	) -> None:
		connection.execute(
			"""
			INSERT INTO artifacts (session_id, frame_index, artifact_type, path, metadata_json)
			VALUES (?, ?, ?, ?, ?)
			""",
			(
				session_id,
				frame_index,
				artifact.artifact_type,
				str(artifact.path),
				_json_dumps(artifact.metadata),
			),
		)


__all__ = [
	"DEFAULT_OPERATIONAL_DB_PATH",
	"OperationalLogStore",
	"connect",
	"initialize",
]
