"""Target-specific classifier adapters.

The mock classifiers expose the same target-by-target prediction shape the
future real binary classifiers should use: one probability-like score per
target after preprocessing.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from app.core.config import SETTINGS
from app.core.paths import ASSET_WEIGHTS_DIR
from app.service.operational_types import ClassifierResult, PreprocessResult, TARGET_LABELS, TargetScore
from app.service.target_service import TargetModelHandle, TargetService


class TargetClassifierAdapter(Protocol):
	target_name: str

	def predict_bgr(
		self,
		image_bgr: np.ndarray,
		*,
		normalized_chw: np.ndarray | None = None,
		scenario: str = "normal",
		frame_index: int = 0,
		variant_name: str = "base",
	) -> TargetScore:
		"""Return this target classifier's score for one preprocessed image."""


SCENARIO_SCORE_TABLE: dict[str, dict[str, float]] = {
	"normal_target2_accept": {"Target1": 0.12, "Target2": 0.91, "Target3": 0.18, "Target4": 0.16},
	"single_pass_accept": {"Target1": 0.98, "Target2": 0.28, "Target3": 0.24, "Target4": 0.22},
	"multi_pass_clear_winner": {"Target1": 0.98, "Target2": 0.80, "Target3": 0.31, "Target4": 0.25},
	"unknown_no_pass": {"Target1": 0.24, "Target2": 0.38, "Target3": 0.27, "Target4": 0.25},
	"ambiguous_reinspect": {"Target1": 0.20, "Target2": 0.91, "Target3": 0.92, "Target4": 0.20},
	"poor_detector_reinspect": {"Target1": 0.16, "Target2": 0.90, "Target3": 0.20, "Target4": 0.18},
	"no_detection": {"Target1": 0.18, "Target2": 0.22, "Target3": 0.20, "Target4": 0.19},
}

VARIANT_OFFSETS: dict[str, dict[str, float]] = {
	"base": {},
	"expanded_crop": {"Target2": -0.01, "Target3": 0.01},
	"deskew_off": {"Target2": 0.00, "Target3": 0.01},
	"resize_variant": {"Target2": 0.01, "Target3": -0.01},
}

SCENARIO_ALIASES = {
	"normal": "normal_target2_accept",
	"single-pass": "single_pass_accept",
	"single_pass": "single_pass_accept",
	"no-pass": "unknown_no_pass",
	"no_pass": "unknown_no_pass",
	"multi-pass-clear": "multi_pass_clear_winner",
	"multi_pass_clear": "multi_pass_clear_winner",
	"multi-pass-ambiguous": "ambiguous_reinspect",
	"multi_pass_ambiguous": "ambiguous_reinspect",
}


def scenario_scores(scenario: str, *, frame_index: int, variant_name: str) -> dict[str, float]:
	normalized = scenario.strip().lower()
	normalized = SCENARIO_ALIASES.get(normalized, normalized)
	base_scores = dict(SCENARIO_SCORE_TABLE.get(normalized, SCENARIO_SCORE_TABLE["normal_target2_accept"]))
	offsets = VARIANT_OFFSETS.get(variant_name, {})
	frame_jitter = ((frame_index % 5) - 2) * 0.002

	if normalized == "ambiguous_reinspect":
		frame_jitter = 0.0

	for target_name in TARGET_LABELS:
		score = base_scores[target_name] + offsets.get(target_name, 0.0) + frame_jitter
		base_scores[target_name] = round(float(min(0.999, max(0.0, score))), 4)
	return base_scores


class MockTargetClassifier:
	"""One mock binary classifier for a specific target."""

	def __init__(self, target_name: str) -> None:
		self.target_name = target_name
		self.classifier_name = f"mock_{target_name.lower()}_classifier"

	def predict_bgr(
		self,
		image_bgr: np.ndarray,
		*,
		normalized_chw: np.ndarray | None = None,
		scenario: str = "normal",
		frame_index: int = 0,
		variant_name: str = "base",
	) -> TargetScore:
		scores = scenario_scores(scenario, frame_index=frame_index, variant_name=variant_name)
		return TargetScore(
			target_name=self.target_name,
			score=scores[self.target_name],
			classifier_name=self.classifier_name,
			scenario=scenario,
		)


class MockTargetClassifierSuite:
	"""Run the four target-specific mock classifiers."""

	def __init__(self, target_names: tuple[str, ...] = TARGET_LABELS) -> None:
		self.classifiers = {target_name: MockTargetClassifier(target_name) for target_name in target_names}

	def predict_all(
		self,
		preprocess_result: PreprocessResult,
		*,
		scenario: str = "normal",
		frame_index: int = 0,
	) -> ClassifierResult:
		predictions: dict[str, TargetScore] = {}
		for target_name, classifier in self.classifiers.items():
			predictions[target_name] = classifier.predict_bgr(
				preprocess_result.processed_bgr,
				normalized_chw=preprocess_result.normalized_chw,
				scenario=scenario,
				frame_index=frame_index,
				variant_name=preprocess_result.variant.name,
			)
		scores = {target_name: prediction.score for target_name, prediction in predictions.items()}
		return ClassifierResult(scores=scores, predictions=predictions, scenario=scenario)


TARGET_TO_REPO_NAME = {
	"Target1": "target_1",
	"Target2": "target_2",
	"Target3": "target_3",
	"Target4": "target_4",
}


class RealTargetClassifier:
	"""Adapter around the repository's target-specific real classifier loader."""

	def __init__(
		self,
		target_name: str,
		*,
		target_service: TargetService,
		model_input_size: int = SETTINGS.operational.PREPROCESS_OUTPUT_SIZE,
	) -> None:
		self.target_name = target_name
		self.repo_target_name = TARGET_TO_REPO_NAME[target_name]
		self.target_service = target_service
		self.model_input_size = model_input_size
		self.classifier_name = f"real_{self.repo_target_name}_classifier"
		self._handle: TargetModelHandle | None = None

	def handle(self) -> TargetModelHandle:
		if self._handle is None:
			self._handle = self.target_service.get_handle(self.repo_target_name)
		return self._handle

	@property
	def backend(self) -> str:
		return self.handle().backend

	@property
	def weights(self) -> str:
		return str(self.handle().weights)

	def predict_bgr(
		self,
		image_bgr: np.ndarray,
		*,
		normalized_chw: np.ndarray | None = None,
		scenario: str = "normal",
		frame_index: int = 0,
		variant_name: str = "base",
	) -> TargetScore:
		model_input_bgr = image_bgr
		height, width = image_bgr.shape[:2]
		if height != self.model_input_size or width != self.model_input_size:
			model_input_bgr = cv2.resize(image_bgr, (self.model_input_size, self.model_input_size))
		prediction = self.handle().predict_bgr(
			model_input_bgr,
			device=self.target_service.device,
			threshold=0.5,
		)
		return TargetScore(
			target_name=self.target_name,
			score=round(float(prediction.prob_yes), 4),
			classifier_name=f"{self.classifier_name}:{self.backend}",
			scenario=scenario,
		)


class RealTargetClassifierSuite:
	"""Run the four real target-specific binary classifiers."""

	def __init__(
		self,
		target_names: tuple[str, ...] = TARGET_LABELS,
		*,
		target_root=ASSET_WEIGHTS_DIR,
		device: str = SETTINGS.sequence.device,
		prefer_openvino: bool = True,
	) -> None:
		self.target_service = TargetService(
			target_root=target_root,
			device=device,
			prefer_openvino=prefer_openvino,
		)
		self.classifiers = {
			target_name: RealTargetClassifier(target_name, target_service=self.target_service)
			for target_name in target_names
		}

	def predict_all(
		self,
		preprocess_result: PreprocessResult,
		*,
		scenario: str = "normal",
		frame_index: int = 0,
	) -> ClassifierResult:
		predictions: dict[str, TargetScore] = {}
		for target_name, classifier in self.classifiers.items():
			predictions[target_name] = classifier.predict_bgr(
				preprocess_result.processed_bgr,
				normalized_chw=preprocess_result.normalized_chw,
				scenario=scenario,
				frame_index=frame_index,
				variant_name=preprocess_result.variant.name,
			)
		scores = {target_name: prediction.score for target_name, prediction in predictions.items()}
		return ClassifierResult(scores=scores, predictions=predictions, scenario=scenario)

	def backends_used(self) -> dict[str, str]:
		return {target_name: classifier.backend for target_name, classifier in self.classifiers.items()}

	def weights_used(self) -> dict[str, str]:
		return {target_name: classifier.weights for target_name, classifier in self.classifiers.items()}


__all__ = [
	"MockTargetClassifier",
	"MockTargetClassifierSuite",
	"RealTargetClassifier",
	"RealTargetClassifierSuite",
	"TargetClassifierAdapter",
	"scenario_scores",
]
