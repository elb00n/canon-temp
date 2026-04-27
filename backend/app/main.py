"""FastAPI entrypoint for the Canon inspection backend.

This file keeps the operational debug routes and mounts the frontend-facing
REST/WebSocket transport routers around the same operational pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
	sys.path.insert(0, str(_BACKEND_ROOT))

from app.api.api_router import router as api_router
from app.api.ws_router import _broadcast_loop, router as ws_router
from app.core.config import SETTINGS
from app.core.paths import ensure_project_dirs
from app.service.operational_db import DEFAULT_OPERATIONAL_DB_PATH, initialize as init_operational_db
from app.service.operational_pipeline import OperationalPipelineConfig, build_synthetic_frame
from app.service.operational_runtime import (
	decode_image_bytes,
	get_operational_service,
	load_video_frames,
	loaded_model_modes,
)


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
	title="Canon Backend Operational API",
	description="Frontend transport plus operational inference/debug API",
	version="0.1.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:5173",
		"http://127.0.0.1:5173",
		"http://localhost:3000",
		"http://127.0.0.1:3000",
	],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


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


@app.on_event("startup")
async def on_startup() -> None:
	ensure_project_dirs()

	from db.database import DB_PATH, initialize as init_compat_db

	init_compat_db(DB_PATH)
	init_operational_db(DEFAULT_OPERATIONAL_DB_PATH)
	asyncio.create_task(_broadcast_loop())

	def _warmup() -> None:
		service = get_operational_service("real")
		dummy = build_synthetic_frame(scenario="normal_target1_accept")
		service.infer_image(
			dummy,
			scenario="normal_target1_accept",
			session_id="__warmup__",
			reset_state=True,
			save_artifacts=False,
		)

	logger.info("Warming up models on GPU...")
	await asyncio.to_thread(_warmup)
	logger.info("Canon backend ready. Operational DB: %s", DEFAULT_OPERATIONAL_DB_PATH)


@app.on_event("shutdown")
async def on_shutdown() -> None:
	logger.info("Canon backend shutting down.")


def _http_error(exc: Exception) -> HTTPException:
	if isinstance(exc, FileNotFoundError):
		return HTTPException(status_code=404, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=400, detail=str(exc))
	return HTTPException(status_code=500, detail=str(exc))


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
		"loaded_model_modes": loaded_model_modes(),
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
	service = get_operational_service(model_mode)
	try:
		image_bgr = build_synthetic_frame(scenario=scenario) if file is None else decode_image_bytes(await file.read())
		result = service.infer_image(
			image_bgr,
			scenario=scenario,
			session_id=session_id,
			strict_sequence=strict_sequence,
			reset_state=reset_state,
			save_artifacts=save_artifacts,
		)
	except Exception as exc:
		raise _http_error(exc) from exc

	response = result.response_dict()
	response["db_path"] = str(service.config.db_path)
	return response


@app.post("/infer/sequence")
def infer_sequence(request: SequenceInferenceRequest) -> dict[str, object]:
	service = get_operational_service(request.model_mode)
	try:
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
	except Exception as exc:
		raise _http_error(exc) from exc


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
	service = get_operational_service(model_mode)
	try:
		return service.run_smoke_scenario(
			scenario,
			width=width,
			height=height,
			mode=mode,
			frame_count=frame_count,
		)
	except Exception as exc:
		raise _http_error(exc) from exc


if __name__ == "__main__":
	uvicorn.run(
		"app.main:app",
		host="0.0.0.0",
		port=8080,
		reload=True,
		log_level="info",
	)


__all__ = ["app"]
