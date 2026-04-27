"""WebSocket transport for dashboard streaming.

Incoming frames are decoded and sent through the operational pipeline. The
frontend receives the same video-frame broadcasts as before plus state payloads
mapped from operational decisions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.payload_mapper import operational_result_to_frontend_payload
from app.service.operational_runtime import get_operational_service, normalize_model_mode


logger = logging.getLogger(__name__)
router = APIRouter()

FRAME_BROADCAST_INTERVAL = 0.1
STATE_BROADCAST_INTERVAL = 0.5
INFERENCE_MIN_INTERVAL = 0.35
JPEG_QUALITY = 90
CAM_WS_SOURCE = "CAM_WS"


class ConnectionManager:
	def __init__(self) -> None:
		self._clients: list[WebSocket] = []
		self._lock = asyncio.Lock()

	async def connect(self, ws: WebSocket) -> None:
		await ws.accept()
		async with self._lock:
			self._clients.append(ws)

	async def disconnect(self, ws: WebSocket) -> None:
		async with self._lock:
			self._clients = [client for client in self._clients if client is not ws]

	async def broadcast(self, data: str) -> None:
		async with self._lock:
			dead: list[WebSocket] = []
			for client in self._clients:
				try:
					await client.send_text(data)
				except Exception:
					dead.append(client)
			for client in dead:
				self._clients = [candidate for candidate in self._clients if candidate is not client]

	def count(self) -> int:
		return len(self._clients)


manager = ConnectionManager()


def _safe_id_part(value: str) -> str:
	return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:80]


@dataclass
class CameraState:
	cam_id: str
	session_id: str
	predicted_label: str = "unknown"
	final_label: str = "unknown"
	confidence: float = 0.0
	confirmed_state: str = "Idle"
	is_unknown: bool = False
	ambiguous: bool = False
	reinspect_needed: bool = False
	inference: bool = False
	current_step_index: int = 1
	allowed_transition: bool = True
	system_message: str = "WAITING..."
	last_frame: Any = field(default=None, repr=False)
	last_frame_jpeg: str | None = field(default=None, repr=False)
	last_frame_time: float = 0.0
	last_inference_time: float = 0.0
	inference_running: bool = False
	frame_index: int = 0
	latest_payload: dict[str, Any] | None = field(default=None, repr=False)

	def to_payload(self) -> dict[str, Any]:
		if self.latest_payload is not None:
			return {"cameraId": self.cam_id, "payload": self.latest_payload}
		return {
			"cameraId": self.cam_id,
			"payload": {
				"predicted_label": self.predicted_label,
				"final_label": self.final_label,
				"confidence": self.confidence,
				"confirmed_state": self.confirmed_state,
				"is_unknown": self.is_unknown,
				"ambiguous": self.ambiguous,
				"reinspect_needed": self.reinspect_needed,
				"inference": self.inference,
				"logic": {
					"current_step_index": self.current_step_index,
					"confirmed_state": self.confirmed_state,
					"allowed_transition": self.allowed_transition,
					"confidence": self.confidence,
				},
				"display": {
					"system_message": self.system_message,
				},
			},
		}


class CameraStateStore:
	def __init__(self) -> None:
		self._cameras: dict[str, CameraState] = {}
		self._lock = threading.Lock()

	def _new_state(self, cam_id: str) -> CameraState:
		session_id = f"ws_{_safe_id_part(cam_id)}_{int(time.time() * 1000)}"
		return CameraState(cam_id=cam_id, session_id=session_id)

	def register(self, cam_id: str, *, reset_session: bool = False) -> CameraState:
		with self._lock:
			if reset_session or cam_id not in self._cameras:
				self._cameras[cam_id] = self._new_state(cam_id)
			return self._cameras[cam_id]

	def unregister(self, cam_id: str) -> None:
		with self._lock:
			self._cameras.pop(cam_id, None)

	def update_frame(self, cam_id: str, frame_bgr: Any, frame_jpeg: str | None = None) -> None:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state is None:
				state = self._new_state(cam_id)
				self._cameras[cam_id] = state
			state.last_frame = frame_bgr.copy()
			state.last_frame_jpeg = frame_jpeg
			state.last_frame_time = time.monotonic()

	def prepare_inference(
		self,
		cam_id: str,
		*,
		now: float,
		min_interval: float,
		session_id: str | None = None,
		reset_session: bool = False,
	) -> tuple[str, int, bool, bool]:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state is None or reset_session:
				state = self._new_state(cam_id)
				self._cameras[cam_id] = state
			if session_id:
				if state.session_id != session_id:
					state.frame_index = 0
				state.session_id = session_id
			if state.inference_running:
				return state.session_id, state.frame_index, False, False
			if now - state.last_inference_time < min_interval:
				return state.session_id, state.frame_index, False, False
			frame_index = state.frame_index
			state.frame_index += 1
			state.last_inference_time = now
			state.inference_running = True
			return state.session_id, frame_index, frame_index == 0 or reset_session, True

	def finish_inference(self, cam_id: str, session_id: str) -> None:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state is not None and state.session_id == session_id:
				state.inference_running = False

	def update_payload(self, cam_id: str, payload: dict[str, Any]) -> None:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state is None:
				state = self._new_state(cam_id)
				self._cameras[cam_id] = state
			self._apply_payload(state, payload)

	def update_payload_if_current(self, cam_id: str, session_id: str, payload: dict[str, Any]) -> bool:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state is None or state.session_id != session_id:
				return False
			self._apply_payload(state, payload)
			return True

	def _apply_payload(self, state: CameraState, payload: dict[str, Any]) -> None:
		state.latest_payload = payload
		state.predicted_label = str(payload.get("predicted_label", "unknown"))
		state.final_label = str(payload.get("final_label", "unknown"))
		state.confidence = float(payload.get("confidence") or 0.0)
		state.confirmed_state = str(payload.get("confirmed_state", "Analyzing"))
		state.is_unknown = bool(payload.get("is_unknown"))
		state.ambiguous = bool(payload.get("ambiguous"))
		state.reinspect_needed = bool(payload.get("reinspect_needed"))
		state.inference = bool(payload.get("inference", True))
		state.allowed_transition = bool(payload.get("allowed_transition"))
		logic = payload.get("logic")
		if isinstance(logic, dict):
			state.current_step_index = int(logic.get("current_step_index") or state.current_step_index)
		display = payload.get("display")
		if isinstance(display, dict):
			state.system_message = str(display.get("system_message") or state.system_message)

	def update_inference(self, cam_id: str, **kwargs: Any) -> None:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state:
				for key, value in kwargs.items():
					if hasattr(state, key):
						setattr(state, key, value)

	def camera_ids(self) -> list[str]:
		with self._lock:
			return list(self._cameras.keys())

	def get_all_payloads(self) -> list[dict[str, Any]]:
		with self._lock:
			return [state.to_payload() for state in self._cameras.values()]

	def get_frame(self, cam_id: str) -> Any:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state and state.last_frame is not None:
				return state.last_frame.copy()
			return None

	def get_frame_jpeg(self, cam_id: str) -> str | None:
		with self._lock:
			state = self._cameras.get(cam_id)
			if state and state.last_frame_jpeg:
				return state.last_frame_jpeg
			return None


camera_store = CameraStateStore()


def _encode_frame(frame_bgr: Any, quality: int = JPEG_QUALITY) -> str:
	import cv2

	_, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
	return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_frame_text(data: str) -> Any:
	import cv2
	import numpy as np

	if "," in data:
		data = data.split(",", 1)[1]
	raw = base64.b64decode(data)
	array = np.frombuffer(raw, np.uint8)
	frame_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
	if frame_bgr is None:
		raise ValueError("failed to decode websocket frame")
	return frame_bgr


def _frame_text_to_base64(data: str) -> str:
	if "," in data:
		return data.split(",", 1)[1]
	return data


def _parse_target_order(value: Any) -> list[str] | None:
	if value is None:
		return None
	if isinstance(value, list):
		return [str(item) for item in value if str(item).strip()]
	if isinstance(value, str):
		return [part.strip() for part in value.split(",") if part.strip()]
	return None


def _parse_bool(value: Any, *, default: bool = False) -> bool:
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	if isinstance(value, str):
		text = value.strip().lower()
		if text in {"1", "true", "yes", "y", "on"}:
			return True
		if text in {"0", "false", "no", "n", "off"}:
			return False
	return default


def _extract_frame_envelope(raw_text: str, *, default_cam_id: str) -> dict[str, Any]:
	try:
		message = json.loads(raw_text)
	except json.JSONDecodeError:
		return {"camera_id": default_cam_id, "frame": raw_text}

	if not isinstance(message, dict):
		return {"camera_id": default_cam_id, "frame": raw_text}

	frame = message.get("frame") or message.get("image") or message.get("data")
	camera_id = str(message.get("cameraId") or message.get("cam_id") or default_cam_id)
	return {
		"camera_id": camera_id,
		"frame": frame,
		"scenario": message.get("scenario"),
		"model_mode": message.get("model_mode"),
		"session_id": message.get("session_id"),
		"target_order": message.get("target_order"),
		"strict_sequence": message.get("strict_sequence"),
		"reset_state": message.get("reset_state"),
		"save_artifacts": message.get("save_artifacts"),
	}


def _infer_frame_sync(
	frame_bgr: Any,
	*,
	model_mode: str,
	scenario: str,
	session_id: str,
	frame_index: int,
	target_order: list[str] | None,
	strict_sequence: bool,
	reset_state: bool,
	save_artifacts: bool | None,
) -> dict[str, Any]:
	service = get_operational_service(model_mode)
	result = service.infer_image(
		frame_bgr,
		scenario=scenario,
		session_id=session_id,
		frame_index=frame_index,
		target_order=target_order,
		strict_sequence=strict_sequence,
		reset_state=reset_state,
		save_artifacts=save_artifacts,
	)
	response = result.response_dict()
	response["db_path"] = str(service.config.db_path)
	return response


async def _process_frame_envelope(
	envelope: dict[str, Any],
	*,
	default_model_mode: str,
	default_scenario: str,
	default_strict_sequence: bool = True,
) -> dict[str, Any] | None:
	frame_text = envelope.get("frame")
	if not isinstance(frame_text, str) or not frame_text:
		return None

	camera_id = str(envelope.get("camera_id") or CAM_WS_SOURCE)
	frame_bgr = _decode_frame_text(frame_text)
	camera_store.register(camera_id)
	camera_store.update_frame(camera_id, frame_bgr, _frame_text_to_base64(frame_text))

	model_mode = normalize_model_mode(str(envelope.get("model_mode") or default_model_mode))
	scenario = str(envelope.get("scenario") or default_scenario)
	session_id, frame_index, first_frame, should_infer = camera_store.prepare_inference(
		camera_id,
		now=time.monotonic(),
		min_interval=INFERENCE_MIN_INTERVAL,
		session_id=str(envelope.get("session_id") or "") or None,
		reset_session=_parse_bool(envelope.get("reset_state")),
	)
	if not should_infer:
		return None

	loop = asyncio.get_running_loop()
	try:
		response = await loop.run_in_executor(
			None,
			partial(
				_infer_frame_sync,
				frame_bgr,
				model_mode=model_mode,
				scenario=scenario,
				session_id=session_id,
				frame_index=frame_index,
				target_order=_parse_target_order(envelope.get("target_order")),
				strict_sequence=_parse_bool(envelope.get("strict_sequence"), default=default_strict_sequence),
				reset_state=first_frame,
				save_artifacts=envelope.get("save_artifacts"),
			),
		)
		payload = operational_result_to_frontend_payload(response)
	except Exception as exc:
		logger.exception("stream inference failed for %s", camera_id)
		payload = {
			"predicted_label": "error",
			"final_label": "unknown",
			"confidence": 0.0,
			"confirmed_state": "Error",
			"allowed_transition": False,
			"inference": True,
			"is_unknown": True,
			"ambiguous": False,
			"reinspect_needed": False,
			"logic": {
				"current_step_index": 1,
				"confirmed_state": "Error",
				"allowed_transition": False,
				"confidence": 0.0,
				"error": str(exc),
			},
			"display": {"system_message": "BACKEND ERROR"},
		}
	finally:
		camera_store.finish_inference(camera_id, session_id)

	if camera_store.update_payload_if_current(camera_id, session_id, payload):
		await manager.broadcast(json.dumps([{"cameraId": camera_id, "payload": payload}]))
	return payload


async def _run_inference_task(
	*,
	camera_id: str,
	frame_bgr: Any,
	model_mode: str,
	scenario: str,
	session_id: str,
	frame_index: int,
	target_order: list[str] | None,
	strict_sequence: bool,
	reset_state: bool,
	save_artifacts: bool | None,
	reply_ws: WebSocket | None = None,
) -> None:
	try:
		loop = asyncio.get_running_loop()
		response = await loop.run_in_executor(
			None,
			partial(
				_infer_frame_sync,
				frame_bgr,
				model_mode=model_mode,
				scenario=scenario,
				session_id=session_id,
				frame_index=frame_index,
				target_order=target_order,
				strict_sequence=strict_sequence,
				reset_state=reset_state,
				save_artifacts=save_artifacts,
			),
		)
		payload = operational_result_to_frontend_payload(response)
	except Exception as exc:
		logger.exception("stream inference failed for %s", camera_id)
		payload = {
			"predicted_label": "error",
			"final_label": "unknown",
			"confidence": 0.0,
			"confirmed_state": "Error",
			"allowed_transition": False,
			"inference": True,
			"is_unknown": True,
			"ambiguous": False,
			"reinspect_needed": False,
			"logic": {
				"current_step_index": 1,
				"confirmed_state": "Error",
				"allowed_transition": False,
				"confidence": 0.0,
				"error": str(exc),
			},
			"display": {"system_message": "BACKEND ERROR"},
		}
	finally:
		camera_store.finish_inference(camera_id, session_id)

	if not camera_store.update_payload_if_current(camera_id, session_id, payload):
		return

	await manager.broadcast(json.dumps([{"cameraId": camera_id, "payload": payload}]))
	if reply_ws is not None:
		try:
			await reply_ws.send_text(json.dumps({"type": "inference", "cameraId": camera_id, "payload": payload}))
		except Exception:
			pass


async def _enqueue_frame_envelope(
	envelope: dict[str, Any],
	*,
	default_model_mode: str,
	default_scenario: str,
	default_strict_sequence: bool = True,
	reply_ws: WebSocket | None = None,
) -> bool:
	frame_text = envelope.get("frame")
	if not isinstance(frame_text, str) or not frame_text:
		return False

	camera_id = str(envelope.get("camera_id") or CAM_WS_SOURCE)
	frame_bgr = _decode_frame_text(frame_text)
	camera_store.register(camera_id)
	camera_store.update_frame(camera_id, frame_bgr, _frame_text_to_base64(frame_text))

	model_mode = normalize_model_mode(str(envelope.get("model_mode") or default_model_mode))
	scenario = str(envelope.get("scenario") or default_scenario)
	session_id, frame_index, first_frame, should_infer = camera_store.prepare_inference(
		camera_id,
		now=time.monotonic(),
		min_interval=INFERENCE_MIN_INTERVAL,
		session_id=str(envelope.get("session_id") or "") or None,
		reset_session=_parse_bool(envelope.get("reset_state")),
	)
	if not should_infer:
		return False

	asyncio.create_task(
		_run_inference_task(
			camera_id=camera_id,
			frame_bgr=frame_bgr.copy(),
			model_mode=model_mode,
			scenario=scenario,
			session_id=session_id,
			frame_index=frame_index,
			target_order=_parse_target_order(envelope.get("target_order")),
			strict_sequence=_parse_bool(envelope.get("strict_sequence"), default=default_strict_sequence),
			reset_state=first_frame,
			save_artifacts=envelope.get("save_artifacts"),
			reply_ws=reply_ws,
		)
	)
	return True


async def _broadcast_loop() -> None:
	last_state_time = 0.0

	while True:
		await asyncio.sleep(FRAME_BROADCAST_INTERVAL)

		if manager.count() == 0:
			continue

		cam_ids = camera_store.camera_ids()
		now = time.monotonic()

		for cam_id in cam_ids:
			frame_jpeg = camera_store.get_frame_jpeg(cam_id)
			if frame_jpeg is None:
				frame = camera_store.get_frame(cam_id)
				if frame is None:
					continue
				frame_jpeg = _encode_frame(frame)
			try:
				await manager.broadcast(
					json.dumps({"type": "video_frame", "cameraId": cam_id, "frame": frame_jpeg})
				)
			except Exception as exc:
				logger.warning("frame broadcast error for %s: %s", cam_id, exc)

		if now - last_state_time >= STATE_BROADCAST_INTERVAL:
			last_state_time = now
			if cam_ids:
				await manager.broadcast(json.dumps({"type": "camera_list", "cameras": cam_ids}))
			payloads = camera_store.get_all_payloads()
			if payloads:
				await manager.broadcast(json.dumps(payloads))


@router.websocket("/ws")
async def ws_main(ws: WebSocket) -> None:
	await manager.connect(ws)
	logger.info("WS client connected. total=%d", manager.count())

	cam_ids = camera_store.camera_ids()
	if cam_ids:
		await ws.send_text(json.dumps({"type": "camera_list", "cameras": cam_ids}))

	default_model_mode = ws.query_params.get("model_mode", "real")
	default_scenario = ws.query_params.get("scenario", "normal_target2_accept")
	default_strict_sequence = _parse_bool(ws.query_params.get("strict_sequence"), default=True)

	try:
		while True:
			raw_text = await ws.receive_text()
			envelope = _extract_frame_envelope(raw_text, default_cam_id=CAM_WS_SOURCE)
			if envelope.get("frame"):
				await _enqueue_frame_envelope(
					envelope,
					default_model_mode=default_model_mode,
					default_scenario=default_scenario,
					default_strict_sequence=default_strict_sequence,
				)
	except WebSocketDisconnect:
		pass
	finally:
		await manager.disconnect(ws)
		logger.info("WS client disconnected. total=%d", manager.count())


@router.websocket("/ws/source")
async def ws_source(ws: WebSocket) -> None:
	await ws.accept()

	cam_id = ws.query_params.get("cameraId") or ws.query_params.get("cam_id") or CAM_WS_SOURCE
	default_model_mode = ws.query_params.get("model_mode", "real")
	default_scenario = ws.query_params.get("scenario", "normal_target2_accept")
	default_strict_sequence = _parse_bool(ws.query_params.get("strict_sequence"), default=True)
	camera_store.register(cam_id, reset_session=True)

	await manager.broadcast(json.dumps({"type": "camera_list", "cameras": camera_store.camera_ids()}))
	logger.info("Mobile source connected as %s", cam_id)

	try:
		while True:
			raw_text = await ws.receive_text()
			envelope = _extract_frame_envelope(raw_text, default_cam_id=cam_id)
			await _enqueue_frame_envelope(
				envelope,
				default_model_mode=default_model_mode,
				default_scenario=default_scenario,
				default_strict_sequence=default_strict_sequence,
				reply_ws=ws,
			)
	except WebSocketDisconnect:
		pass
	finally:
		camera_store.unregister(cam_id)
		await manager.broadcast(json.dumps({"type": "camera_list", "cameras": camera_store.camera_ids()}))
		logger.info("Mobile source disconnected. %s unregistered.", cam_id)


def register_camera(cam_id: str) -> None:
	camera_store.register(cam_id)


def unregister_camera(cam_id: str) -> None:
	camera_store.unregister(cam_id)


def push_frame(cam_id: str, frame_bgr: Any) -> None:
	camera_store.register(cam_id)
	camera_store.update_frame(cam_id, frame_bgr)


def push_inference_state(cam_id: str, **kwargs: Any) -> None:
	camera_store.update_inference(cam_id, **kwargs)


__all__ = [
	"CAM_WS_SOURCE",
	"_broadcast_loop",
	"camera_store",
	"manager",
	"push_frame",
	"push_inference_state",
	"register_camera",
	"router",
	"unregister_camera",
]
