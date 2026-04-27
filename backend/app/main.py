"""FastAPI entrypoint for operational smoke inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import SETTINGS
from app.service.operational_pipeline import OperationalInferenceService, OperationalPipelineConfig, build_synthetic_frame


app = FastAPI(title="Canon Backend Operational API", version="0.1.0")
SERVICES: dict[str, OperationalInferenceService] = {}


def get_service(model_mode: Literal["real", "mock"]) -> OperationalInferenceService:
	if model_mode not in SERVICES:
		SERVICES[model_mode] = OperationalInferenceService(
			OperationalPipelineConfig(model_mode=model_mode),
		)
	return SERVICES[model_mode]


class SequenceInferenceRequest(BaseModel):
	scenario: str = "normal_target2_accept"
	model_mode: Literal["real", "mock"] = "real"
	session_id: str | None = None
	target_order: list[str] | None = None
	strict_sequence: bool = False
	reset_state: bool = True
	save_artifacts: bool | None = None
	frame_count: int = Field(default=5, ge=1, le=120)
	width: int = Field(default=640, ge=64, le=4096)
	height: int = Field(default=480, ge=64, le=4096)
	video_path: str | None = None


def decode_image_bytes(payload: bytes) -> np.ndarray:
	array = np.frombuffer(payload, dtype=np.uint8)
	image = cv2.imdecode(array, cv2.IMREAD_COLOR)
	if image is None:
		raise HTTPException(status_code=400, detail="failed to decode image payload")
	return image


def load_video_frames(video_path: Path, *, max_frames: int) -> list[np.ndarray]:
	if not video_path.exists():
		raise HTTPException(status_code=404, detail=f"video_path not found: {video_path}")
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise HTTPException(status_code=400, detail=f"failed to open video_path: {video_path}")
	frames: list[np.ndarray] = []
	try:
		while len(frames) < max_frames:
			ok, frame = capture.read()
			if not ok:
				break
			frames.append(frame)
	finally:
		capture.release()
	if not frames:
		raise HTTPException(status_code=400, detail="no frames could be read from video_path")
	return frames


@app.get("/health")
def health() -> dict[str, object]:
	return {
		"ok": True,
		"thresholds": SETTINGS.operational.thresholds,
		"decision_margin": SETTINGS.operational.DECISION_MARGIN,
		"delta_accept": SETTINGS.operational.DELTA_ACCEPT,
		"reinspect_window": SETTINGS.operational.REINSPECT_WINDOW,
		"default_model_mode": "real",
		"switchable_model_modes": ["real", "mock"],
		"loaded_model_modes": sorted(SERVICES),
		"db_path": str(OperationalPipelineConfig().db_path),
	}


@app.post("/infer/image")
async def infer_image(
	scenario: str = "normal_target2_accept",
	session_id: str | None = None,
	strict_sequence: bool = False,
	reset_state: bool = False,
	save_artifacts: bool | None = None,
	model_mode: Literal["real", "mock"] = "real",
	file: UploadFile | None = File(default=None),
) -> dict[str, object]:
	service = get_service(model_mode)
	if file is None:
		image_bgr = build_synthetic_frame(scenario=scenario)
	else:
		image_bgr = decode_image_bytes(await file.read())
	result = service.infer_image(
		image_bgr,
		scenario=scenario,
		session_id=session_id,
		strict_sequence=strict_sequence,
		reset_state=reset_state,
		save_artifacts=save_artifacts,
	)
	response = result.response_dict()
	response["db_path"] = str(service.config.db_path)
	return response


@app.post("/infer/sequence")
def infer_sequence(request: SequenceInferenceRequest) -> dict[str, object]:
	service = get_service(request.model_mode)
	if request.video_path:
		frames = load_video_frames(Path(request.video_path), max_frames=request.frame_count)
	else:
		frames = [
			build_synthetic_frame(
				scenario=request.scenario,
				width=request.width,
				height=request.height,
				frame_index=index,
			)
			for index in range(request.frame_count)
		]
	return service.infer_sequence(
		frames,
		scenario=request.scenario,
		session_id=request.session_id,
		target_order=request.target_order,
		strict_sequence=request.strict_sequence,
		reset_state=request.reset_state,
		save_artifacts=request.save_artifacts,
	)


@app.post("/infer/video")
def infer_video(request: SequenceInferenceRequest) -> dict[str, object]:
	return infer_sequence(request)


@app.get("/smoke/{scenario}")
def smoke_scenario(
	scenario: str,
	mode: Literal["image", "sequence"] = "image",
	model_mode: Literal["real", "mock"] = "mock",
	width: int = 640,
	height: int = 480,
	frame_count: int = 5,
) -> dict[str, object]:
	service = get_service(model_mode)
	return service.run_smoke_scenario(
		scenario,
		width=width,
		height=height,
		mode=mode,
		frame_count=frame_count,
	)


__all__ = ["app"]
