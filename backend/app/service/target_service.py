"""Target classification service layer.

This module centralizes target weight resolution, backend selection
(OpenVINO first, PyTorch fallback), and cached prediction helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.config import SETTINGS
from app.core.paths import ASSET_WEIGHTS_DIR, asset_openvino_model_file, asset_weight_file
from app.models.target_model import BinaryPrediction, DEFAULT_THRESHOLD, load_openvino_target_model, load_target_model


BackendName = Literal["openvino", "pytorch"]


@dataclass(slots=True)
class TargetModelHandle:
	"""A loaded target model and the backend used to run it."""

	target_name: str
	backend: BackendName
	weights: Path
	model: object

	def predict_bgr(
		self,
		image_bgr,
		*,
		device: str = SETTINGS.sequence.device,
		threshold: float = DEFAULT_THRESHOLD,
	) -> BinaryPrediction:
		"""Run inference on a BGR image using the loaded backend."""
		if self.backend == "openvino":
			return self.model.predict_bgr(image_bgr, threshold=threshold)
		return self.model.predict_bgr(image_bgr, device=device, threshold=threshold)


def resolve_target_weight_path(target_name: str, target_root: Path = ASSET_WEIGHTS_DIR) -> Path:
	"""Resolve the canonical PyTorch weight path for a target."""
	candidates = [
		asset_weight_file(target_name),
		target_root / target_name / "best.pt",
		target_root / target_name / "weights" / "best.pt",
		target_root / f"{target_name}_weight" / "best.pt",
		target_root / f"{target_name}_weight" / "weights" / "best.pt",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		f"target model weights not found for {target_name}; checked: "
		+ ", ".join(str(candidate) for candidate in candidates)
	)


def resolve_target_openvino_model_path(target_name: str, target_root: Path = ASSET_WEIGHTS_DIR) -> Path | None:
	"""Resolve the OpenVINO XML path for a target if it exists."""
	candidates = [
		target_root / target_name / f"{target_name}_weight_openvino" / "model.xml",
		target_root / f"{target_name}_weight_openvino" / "model.xml",
		asset_openvino_model_file(target_name),
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return None


def load_target_model_handle(
	target_name: str,
	*,
	target_root: Path = ASSET_WEIGHTS_DIR,
	device: str = SETTINGS.sequence.device,
	prefer_openvino: bool = True,
) -> TargetModelHandle:
	"""Load a target model using OpenVINO IR when available, otherwise PyTorch."""
	if prefer_openvino:
		openvino_model_path = resolve_target_openvino_model_path(target_name, target_root=target_root)
		if openvino_model_path is not None:
			return TargetModelHandle(
				target_name=target_name,
				backend="openvino",
				weights=openvino_model_path,
				model=load_openvino_target_model(openvino_model_path, device="CPU"),
			)

	weight_path = resolve_target_weight_path(target_name, target_root=target_root)
	return TargetModelHandle(
		target_name=target_name,
		backend="pytorch",
		weights=weight_path,
		model=load_target_model(weight_path, device=device),
	)


class TargetService:
	"""Small cache-backed service for target inference."""

	def __init__(self, *, target_root: Path = ASSET_WEIGHTS_DIR, device: str = SETTINGS.sequence.device, prefer_openvino: bool = True) -> None:
		self.target_root = target_root
		self.device = device
		self.prefer_openvino = prefer_openvino
		self._cache: dict[str, TargetModelHandle] = {}

	def get_handle(self, target_name: str) -> TargetModelHandle:
		if target_name not in self._cache:
			self._cache[target_name] = load_target_model_handle(
				target_name,
				target_root=self.target_root,
				device=self.device,
				prefer_openvino=self.prefer_openvino,
			)
		return self._cache[target_name]

	def predict_bgr(self, target_name: str, image_bgr, *, threshold: float = DEFAULT_THRESHOLD) -> BinaryPrediction:
		handle = self.get_handle(target_name)
		return handle.predict_bgr(image_bgr, device=self.device, threshold=threshold)

	def clear_cache(self) -> None:
		self._cache.clear()


def predict_target_bgr(
	target_name: str,
	image_bgr,
	*,
	target_root: Path = ASSET_WEIGHTS_DIR,
	device: str = SETTINGS.sequence.device,
	threshold: float = DEFAULT_THRESHOLD,
	prefer_openvino: bool = True,
) -> BinaryPrediction:
	"""Convenience helper for one-off target predictions."""
	service = TargetService(target_root=target_root, device=device, prefer_openvino=prefer_openvino)
	return service.predict_bgr(target_name, image_bgr, threshold=threshold)


__all__ = [
	"BackendName",
	"TargetModelHandle",
	"TargetService",
	"load_target_model_handle",
	"predict_target_bgr",
	"resolve_target_openvino_model_path",
	"resolve_target_weight_path",
	"asset_openvino_model_file",
	"DEFAULT_THRESHOLD",
	"BinaryPrediction",
	"SETTINGS",
]
