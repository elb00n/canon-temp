"""Shared access to the operational inference pipeline.

API and WebSocket modules use this helper so they do not create separate
inference paths or separate service registries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app.service.operational_pipeline import OperationalInferenceService, OperationalPipelineConfig


ModelMode = Literal["real", "mock"]
_SERVICES: dict[ModelMode, OperationalInferenceService] = {}


def normalize_model_mode(model_mode: str | None) -> ModelMode:
	mode = (model_mode or "real").strip().lower()
	if mode not in {"real", "mock"}:
		raise ValueError(f"unsupported model_mode: {model_mode}")
	return mode  # type: ignore[return-value]


def get_operational_service(model_mode: str | None = "real") -> OperationalInferenceService:
	mode = normalize_model_mode(model_mode)
	if mode not in _SERVICES:
		_SERVICES[mode] = OperationalInferenceService(
			OperationalPipelineConfig(model_mode=mode, prefer_openvino=False),
		)
	return _SERVICES[mode]


def loaded_model_modes() -> list[str]:
	return sorted(_SERVICES)


def decode_image_bytes(payload: bytes) -> np.ndarray:
	array = np.frombuffer(payload, dtype=np.uint8)
	image = cv2.imdecode(array, cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError("failed to decode image payload")
	return image


def load_video_frames(video_path: Path, *, max_frames: int) -> list[np.ndarray]:
	if not video_path.exists():
		raise FileNotFoundError(f"video_path not found: {video_path}")

	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise ValueError(f"failed to open video_path: {video_path}")

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
		raise ValueError("no frames could be read from video_path")
	return frames


__all__ = [
	"ModelMode",
	"decode_image_bytes",
	"get_operational_service",
	"load_video_frames",
	"loaded_model_modes",
	"normalize_model_mode",
]
