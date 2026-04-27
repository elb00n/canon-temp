"""Frontend-facing REST API transport.

These routes keep the dashboard contract, but inference decisions come from
the operational pipeline instead of a second API-local implementation.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.payload_mapper import operational_response_to_log
from app.core.paths import OUTPUTS_DIR
from app.service.operational_db import DEFAULT_OPERATIONAL_DB_PATH
from app.service.operational_runtime import decode_image_bytes, get_operational_service, load_video_frames
from db.database import (
	DB_PATH,
	get_log_by_id,
	get_logs,
	get_sequence_runs as fetch_sequence_runs,
	initialize,
	insert_log,
	update_log,
)


router = APIRouter(prefix="/api")
initialize(DB_PATH)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
DEFAULT_TARGET_ORDER = ["Target1", "Target2", "Target3", "Target4"]


class OverrideRequest(BaseModel):
	cam_id: str
	predicted_label: str | None = None
	confidence: float | None = None
	is_unknown: bool | None = None
	logic: dict[str, Any] | None = None
	display: dict[str, Any] | None = None


def _parse_target_order(raw: str | None) -> list[str] | None:
	if raw is None or not raw.strip():
		return None
	return [part.strip() for part in raw.split(",") if part.strip()]


def _http_error(exc: Exception) -> HTTPException:
	if isinstance(exc, FileNotFoundError):
		return HTTPException(status_code=404, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=400, detail=str(exc))
	return HTTPException(status_code=500, detail=str(exc))


def _connect_operational_db() -> sqlite3.Connection:
	connection = sqlite3.connect(DEFAULT_OPERATIONAL_DB_PATH)
	connection.row_factory = sqlite3.Row
	return connection


def _read_operational_logs(offset: int, limit: int) -> list[dict[str, Any]]:
	if limit <= 0 or not DEFAULT_OPERATIONAL_DB_PATH.exists():
		return []

	try:
		with _connect_operational_db() as connection:
			rows = connection.execute(
				"""
				SELECT
					fr.id,
					fr.timestamp,
					fr.response_json,
					s.mode AS session_mode
				FROM frame_results fr
				LEFT JOIN sessions s ON s.id = fr.session_id
				ORDER BY fr.id DESC
				LIMIT ? OFFSET ?
				""",
				(limit, offset),
			).fetchall()
	except sqlite3.OperationalError:
		return []

	logs: list[dict[str, Any]] = []
	for row in rows:
		try:
			response = json.loads(row["response_json"])
		except Exception:
			continue
		source_type = f"operational_{row['session_mode'] or 'inference'}"
		logs.append(
			operational_response_to_log(
				response,
				log_id=f"op_{row['id']}",
				source_type=source_type,
			)
		)
	return logs


def _read_operational_response(log_id: int) -> dict[str, Any] | None:
	if not DEFAULT_OPERATIONAL_DB_PATH.exists():
		return None
	try:
		with _connect_operational_db() as connection:
			row = connection.execute(
				"SELECT response_json FROM frame_results WHERE id = ?",
				(log_id,),
			).fetchone()
	except sqlite3.OperationalError:
		return None
	if row is None:
		return None
	try:
		return json.loads(row["response_json"])
	except Exception:
		return None


def _artifact_input_path(response: dict[str, Any]) -> Path | None:
	for artifact in response.get("artifact_paths") or []:
		if isinstance(artifact, dict) and artifact.get("artifact_type") == "input_image":
			path = Path(str(artifact.get("path") or ""))
			if path.exists():
				return path
	return None


def _merge_logs(offset: int, limit: int) -> list[dict[str, Any]]:
	window = max(limit + offset, limit, 1)
	compat_logs = get_logs(offset=0, limit=window)
	operational_logs = _read_operational_logs(offset=0, limit=window)
	combined = compat_logs + operational_logs
	combined.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
	return combined[offset : offset + limit]


@router.get("/inspection-logs")
def get_inspection_logs(offset: int = 0, limit: int = 30) -> JSONResponse:
	"""Return dashboard log rows.

	Operational inference rows are read from operational_runs.sqlite3. Legacy
	override/compatibility rows are still read from factory_test.db so the
	frontend test modal remains compatible.
	"""

	return JSONResponse(_merge_logs(offset=offset, limit=limit))


@router.get("/sequence-runs")
def get_sequence_runs(offset: int = 0, limit: int = 50) -> JSONResponse:
	return JSONResponse(fetch_sequence_runs(offset=offset, limit=limit, total_targets=len(DEFAULT_TARGET_ORDER)))


@router.post("/override")
def override_camera(body: OverrideRequest) -> JSONResponse:
	log_id = insert_log(
		source_type="override",
		confirmed_state=str((body.logic or {}).get("confirmed_state", "Override")),
		predicted_label=body.predicted_label or "",
		confidence=float(body.confidence or 0.0),
		anomaly_flag=bool(body.is_unknown),
		cam_id=body.cam_id,
		target_idx=int((body.logic or {}).get("current_step_index", 0)),
		extra={"logic": body.logic, "display": body.display},
	)
	return JSONResponse({"status": "ok", "cam_id": body.cam_id, "log_id": log_id})


@router.post("/inspect-image")
async def inspect_image(
	files: list[UploadFile] = File(...),
	scenario: str = "normal_target2_accept",
	model_mode: Literal["real", "mock"] = "real",
	session_id: str | None = None,
	target_order: str | None = None,
	strict_sequence: bool = False,
	reset_state: bool = True,
	save_artifacts: bool | None = None,
	frame_count: int = 120,
) -> JSONResponse:
	"""Run uploaded images/videos through the operational pipeline.

	The returned shape stays compatible with the test-mode modal, but each
	result is produced and logged by OperationalInferenceService.
	"""

	service = get_operational_service(model_mode)
	parsed_target_order = _parse_target_order(target_order)
	results: list[dict[str, Any]] = []
	loop = asyncio.get_running_loop()

	tmp_parent = OUTPUTS_DIR / "api_uploads_tmp"
	request_tmp_path = tmp_parent / f"canon_inspect_{uuid4().hex}"
	for upload in files:
		filename = Path(upload.filename or "upload").name
		content = await upload.read()
		ext = Path(filename).suffix.lower()
		file_session_id = session_id or f"api_{Path(filename).stem}"

		try:
			if ext in _IMAGE_EXTS:
				image_bgr = decode_image_bytes(content)
				result = await loop.run_in_executor(
					None,
					lambda: service.infer_image(
						image_bgr,
						scenario=scenario,
						session_id=file_session_id,
						target_order=parsed_target_order,
						strict_sequence=strict_sequence,
						reset_state=reset_state,
						save_artifacts=save_artifacts,
					),
				)
				response = result.response_dict()
				response["db_path"] = str(service.config.db_path)
				results.append(
					{
						**operational_response_to_log(
							response,
							source_type="operational_image_upload",
							file_path=filename,
						),
						"operational_detail": response,
					}
				)
			elif ext in _VIDEO_EXTS:
				from app.service.sequence_service import SequenceService, SequenceRunConfig
				from datetime import datetime, timezone
				import json

				request_tmp_path.mkdir(parents=True, exist_ok=True)
				save_path = request_tmp_path / filename
				save_path.write_bytes(content)

				seq_config = SequenceRunConfig(
					source=[save_path],
					target_order=parsed_target_order or ["target_1", "target_2", "target_3", "target_4"],
				)
				seq_service = SequenceService(seq_config)

				sequence_result_summary = await loop.run_in_executor(
					None,
					lambda: seq_service.process_video(save_path, run_root=request_tmp_path),
				)

				# sequence_service 결과를 operational_log 포맷에 맞게 변환
				targets = sequence_result_summary.get("targets", [])
				
				def _get(obj, key, default=None):
					if isinstance(obj, dict):
						return obj.get(key, default)
					return getattr(obj, key, default)

				completed_labels = []
				scores = {}
				target_times = {}
				final_label = "unknown"
				top_score = 0.0

				for t in targets:
					raw_name = _get(t, "target_name", "unknown")
					# "target_1" -> "Target1" 형식으로 변환하여 프론트엔드 호환성 유지
					t_name = raw_name.replace("target_", "Target").replace("unknown", "Unknown")
					
					if _get(t, "completed", False):
						completed_labels.append(t_name)
						# OpenCV가 기록한 실제 영상 타임코드(ms) → MM:SS 변환
						time_ms = _get(t, "confirmed_time_ms", None)
						if time_ms is not None and time_ms > 0:
							total_seconds = int(time_ms / 1000)
							mins = total_seconds // 60
							secs = total_seconds % 60
							target_times[t_name] = f"{mins:02d}:{secs:02d}"
						else:
							target_times[t_name] = "--:--"
					else:
						target_times[t_name] = "--:--"
						
					scores[t_name] = _get(t, "last_score", 0.0)

				if targets:
					raw_final = _get(targets[-1], "last_label", "unknown")
					final_label = raw_final.replace("target_", "Target").replace("unknown", "Unknown")
					top_score = _get(targets[-1], "last_score", 0.0)

				is_completed = bool(sequence_result_summary.get("completed", False))
				ts = datetime.now(timezone.utc).isoformat()
				
				final_response = {
					"timestamp": ts,
					"session_id": file_session_id,
					"frame_index": sequence_result_summary.get("processed_frames", 0),
					"decision_type": "sequence_service",
					"scores": scores,
					"state": {
						"completed_labels": completed_labels,
						"state_machine_allowed": is_completed,
						"event_type": "accepted" if is_completed else "ignored",
						"final_label": final_label,
						"target_times": target_times,
					},
					"decision": {
						"top1_score": top_score
					},
					"db_path": str(service.config.db_path)
				}

				try:
					from db.database import insert_log as db_insert_log, DB_PATH
					n_targets = len(parsed_target_order or ["target_1", "target_2", "target_3", "target_4"])
					completed_count = len(completed_labels)
					db_insert_log(
						source_type="video",
						confirmed_state="accepted" if is_completed else "ignored",
						predicted_label=final_label or "unknown",
						confidence=top_score or 0.0,
						anomaly_flag=not is_completed,
						file_path=filename,
						cam_id="",
						target_idx=completed_count,
						extra=final_response,
						timestamp=ts,
						db_path=DB_PATH,
					)
				except Exception as db_err:
					import traceback
					print(f"DB INSERT ERROR: {db_err}")
					with open("error_log.txt", "a", encoding="utf-8") as f:
						f.write(f"DB INSERT ERROR: {db_err}\n{traceback.format_exc()}\n")

				results.append(
					{
						**operational_response_to_log(
							final_response,
							source_type="operational_video_upload",
							file_path=filename,
						),
						"frame_count": sequence_result_summary.get("processed_frames", 0),
						"operational_detail": sequence_result_summary,
					}
				)
			else:
				results.append(
					{
						"file": filename,
						"error": f"unsupported file type: {ext}",
						"confirmed_state": "Unsupported",
						"anomaly_flag": True,
					}
				)
		except Exception as exc:
			import traceback
			with open("error_log.txt", "a", encoding="utf-8") as f:
				f.write(f"GLOBAL UPLOAD ERROR: {exc}\n{traceback.format_exc()}\n")
			results.append(
				{
					"file": filename,
					"source_type": "operational_upload",
					"error": str(exc),
					"confirmed_state": "Error",
					"anomaly_flag": True,
				}
			)

	return JSONResponse({"status": "ok", "inspections": results})


@router.post("/reinspect-log/{log_id}")
async def reinspect_log(
	log_id: str,
	model_mode: Literal["real", "mock"] = "real",
	scenario: str = "normal_target2_accept",
	save_artifacts: bool | None = None,
) -> JSONResponse:
	"""Re-run inference when the stored row points at an available image.

	If a compatibility row has no real file path, the route still returns the
	stored row so the existing frontend action does not break.
	"""

	service = get_operational_service(model_mode)
	loop = asyncio.get_running_loop()
	force_operational = log_id.startswith("op_")
	numeric_id_text = log_id.removeprefix("op_")
	if not numeric_id_text.isdigit():
		raise HTTPException(status_code=404, detail="log not found")
	numeric_id = int(numeric_id_text)

	if force_operational:
		response = _read_operational_response(numeric_id)
		if response is None:
			raise HTTPException(status_code=404, detail="log not found")
		input_path = _artifact_input_path(response)
		if input_path is None:
			return JSONResponse(
				{
					"status": "success",
					**operational_response_to_log(response, log_id=log_id, source_type="operational_cached"),
				}
			)
		try:
			image_bgr = decode_image_bytes(input_path.read_bytes())
			result = await loop.run_in_executor(
				None,
				lambda: service.infer_image(
					image_bgr,
					scenario=scenario,
					session_id=f"reinspect_{numeric_id}",
					reset_state=True,
					save_artifacts=save_artifacts,
				),
			)
		except Exception as exc:
			raise _http_error(exc) from exc
		new_response = result.response_dict()
		return JSONResponse(
			{
				"status": "success",
				**operational_response_to_log(
					new_response,
					log_id=log_id,
					source_type="operational_reinspect",
					file_path=str(input_path),
				),
				"operational_detail": new_response,
			}
		)

	compat_log = get_log_by_id(numeric_id)
	if compat_log is None:
		response = _read_operational_response(numeric_id)
		if response is None:
			raise HTTPException(status_code=404, detail="log not found")
		return JSONResponse(
			{
				"status": "success",
				**operational_response_to_log(response, log_id=f"op_{numeric_id}", source_type="operational_cached"),
			}
		)

	file_path = Path(str(compat_log.get("file_path") or ""))
	if not file_path.exists():
		return JSONResponse({"status": "success", **compat_log})

	try:
		ext = file_path.suffix.lower()
		if ext in _IMAGE_EXTS:
			image_bgr = decode_image_bytes(file_path.read_bytes())
			result = await loop.run_in_executor(
				None,
				lambda: service.infer_image(
					image_bgr,
					scenario=scenario,
					session_id=f"compat_reinspect_{numeric_id}",
					reset_state=True,
					save_artifacts=save_artifacts,
				),
			)
			response = result.response_dict()
		elif ext in _VIDEO_EXTS:
			frames = load_video_frames(file_path, max_frames=120)
			sequence_result = await loop.run_in_executor(
				None,
				lambda: service.infer_sequence(
					frames,
					scenario=scenario,
					session_id=f"compat_reinspect_{numeric_id}",
					reset_state=True,
					save_artifacts=save_artifacts,
				),
			)
			response = dict(sequence_result["final_result"])
		else:
			return JSONResponse({"status": "success", **compat_log})
	except Exception as exc:
		raise _http_error(exc) from exc

	mapped = operational_response_to_log(response, log_id=numeric_id, source_type="compat_reinspect")
	update_log(
		numeric_id,
		confirmed_state=str(mapped["confirmed_state"]),
		predicted_label=str(mapped["predicted_label"]),
		confidence=float(mapped["confidence"]),
		anomaly_flag=bool(mapped["anomaly_flag"]),
	)
	return JSONResponse({"status": "success", **mapped, "operational_detail": response})
