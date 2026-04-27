"""Detector adapters for operational orchestration.

The interface is kept close to the YOLO warper so real and mock detectors can
be switched without changing the operational pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.models.warping import YoloScreenWarper
from app.service.operational_types import DetectionResult


class ScreenDetectorAdapter(Protocol):
	def detect(self, image_bgr: np.ndarray, *, scenario: str = "normal") -> DetectionResult:
		"""Return the best screen detection for a BGR image."""


class MockScreenDetector:
	"""Deterministic detector that uses the image size to create a center bbox."""

	detector_name = "mock_screen_detector"

	def __init__(self, *, min_side: int = 64) -> None:
		self.min_side = min_side

	@staticmethod
	def _center_bbox(width: int, height: int, *, scale: float = 1.0) -> tuple[float, float, float, float]:
		box_width = max(1.0, width * 0.72 * scale)
		box_height = max(1.0, height * 0.62 * scale)
		x1 = max(0.0, (width - box_width) / 2.0)
		y1 = max(0.0, (height - box_height) / 2.0)
		x2 = min(float(width - 1), x1 + box_width)
		y2 = min(float(height - 1), y1 + box_height)
		return (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))

	def detect(self, image_bgr: np.ndarray, *, scenario: str = "normal") -> DetectionResult:
		height, width = image_bgr.shape[:2]
		normalized_scenario = scenario.strip().lower()
		if normalized_scenario in {"no_detection", "no_valid_detection"} or min(width, height) < self.min_side:
			return DetectionResult(
				screen_detected=False,
				bbox=None,
				detector_confidence=0.0,
				scenario=scenario,
				detector_name=self.detector_name,
			)

		if normalized_scenario in {"poor_detector_reinspect", "poor_detection", "low_quality_detection"}:
			return DetectionResult(
				screen_detected=True,
				bbox=self._center_bbox(width, height, scale=0.82),
				detector_confidence=0.38,
				scenario=scenario,
				detector_name=self.detector_name,
			)

		return DetectionResult(
			screen_detected=True,
			bbox=self._center_bbox(width, height),
			detector_confidence=0.93,
			scenario=scenario,
			detector_name=self.detector_name,
		)


class YoloDetectorAdapter:
	"""Real detector adapter for later use with trained screen weights."""

	detector_name = "yolo_screen_detector"

	def __init__(
		self,
		weights: Path,
		*,
		device: str = "cpu",
		conf: float = 0.25,
		imgsz: int = 640,
		padding_ratio: float = 0.02,
	) -> None:
		self.warper = YoloScreenWarper(
			weights=weights,
			device=device,
			conf=conf,
			imgsz=imgsz,
			padding_ratio=padding_ratio,
			output_size=imgsz,
			classes=[0],
		)

	def detect(self, image_bgr: np.ndarray, *, scenario: str = "normal") -> DetectionResult:
		detections = self.warper.detect(image_bgr)
		if not detections:
			return DetectionResult(
				screen_detected=False,
				bbox=None,
				detector_confidence=0.0,
				scenario=scenario,
				detector_name=self.detector_name,
			)
		best_detection = max(detections, key=lambda item: float(item["confidence"]))
		bbox = tuple(float(value) for value in np.asarray(best_detection["bbox_xyxy"], dtype=np.float32))
		return DetectionResult(
			screen_detected=True,
			bbox=bbox,
			detector_confidence=float(best_detection["confidence"]),
			scenario=scenario,
			detector_name=self.detector_name,
		)


__all__ = [
	"MockScreenDetector",
	"ScreenDetectorAdapter",
	"YoloDetectorAdapter",
]
