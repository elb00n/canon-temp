"""Shared types for the operational smoke/inference pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TARGET_LABELS = ("Target1", "Target2", "Target3", "Target4")
TARGET_ALIASES = {
	"target_1": "Target1",
	"target1": "Target1",
	"Target1": "Target1",
	"target_2": "Target2",
	"target2": "Target2",
	"Target2": "Target2",
	"target_3": "Target3",
	"target3": "Target3",
	"Target3": "Target3",
	"target_4": "Target4",
	"target4": "Target4",
	"Target4": "Target4",
}


def canonical_target_name(target_name: str) -> str:
	return TARGET_ALIASES.get(target_name, target_name)


@dataclass(slots=True)
class DetectionResult:
	screen_detected: bool
	bbox: tuple[float, float, float, float] | None
	detector_confidence: float
	scenario: str = "normal"
	detector_name: str = "mock"

	def as_dict(self) -> dict[str, Any]:
		return {
			"screen_detected": self.screen_detected,
			"bbox": list(self.bbox) if self.bbox is not None else None,
			"detector_confidence": self.detector_confidence,
			"scenario": self.scenario,
			"detector_name": self.detector_name,
		}


@dataclass(frozen=True, slots=True)
class PreprocessVariant:
	name: str
	padding_ratio: float
	deskew: bool
	output_size: int


@dataclass(slots=True)
class PreprocessResult:
	variant: PreprocessVariant
	bbox: tuple[float, float, float, float]
	crop_bgr: np.ndarray
	processed_bgr: np.ndarray
	normalized_chw: np.ndarray
	deskew_attempted: bool
	deskew_applied: bool

	def metadata(self) -> dict[str, Any]:
		return {
			"variant": self.variant.name,
			"padding_ratio": self.variant.padding_ratio,
			"deskew": self.variant.deskew,
			"output_size": self.variant.output_size,
			"deskew_attempted": self.deskew_attempted,
			"deskew_applied": self.deskew_applied,
			"crop_shape": list(self.crop_bgr.shape),
			"processed_shape": list(self.processed_bgr.shape),
			"normalized_shape": list(self.normalized_chw.shape),
		}


@dataclass(slots=True)
class TargetScore:
	target_name: str
	score: float
	classifier_name: str
	scenario: str

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ClassifierResult:
	scores: dict[str, float]
	predictions: dict[str, TargetScore] = field(default_factory=dict)
	scenario: str = "normal"

	def as_dict(self) -> dict[str, Any]:
		return {
			"scores": self.scores,
			"predictions": {target: prediction.as_dict() for target, prediction in self.predictions.items()},
			"scenario": self.scenario,
		}


@dataclass(slots=True)
class DecisionResult:
	scores: dict[str, float]
	thresholds: dict[str, float]
	passed_targets: list[str]
	predicted_label: str
	decision_type: str
	ambiguous: bool
	unknown: bool
	reinspect_needed: bool
	top1_label: str | None
	top1_score: float | None
	top2_label: str | None
	top2_score: float | None
	margin: float | None
	reason: str

	@property
	def accepted(self) -> bool:
		return (
			not self.ambiguous
			and not self.unknown
			and not self.reinspect_needed
			and self.predicted_label in TARGET_LABELS
			and self.decision_type in {"single_pass_accept", "multi_pass_clear_winner"}
		)

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class StateTransition:
	session_id: str
	expected_label: str | None
	effective_label: str
	final_label: str
	state_machine_allowed: bool
	current_index: int
	completed_labels: list[str]
	event_type: str
	reason: str

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ArtifactRecord:
	artifact_type: str
	path: Path
	metadata: dict[str, Any] = field(default_factory=dict)

	def as_dict(self) -> dict[str, Any]:
		return {
			"artifact_type": self.artifact_type,
			"path": str(self.path),
			"metadata": self.metadata,
		}


@dataclass(slots=True)
class FrameInferenceResult:
	session_id: str
	frame_index: int
	timestamp: str
	model_mode: str
	detection: DetectionResult
	decision: DecisionResult
	state: StateTransition
	scores: dict[str, float]
	thresholds: dict[str, float]
	artifacts: list[ArtifactRecord] = field(default_factory=list)
	reinspect_performed: bool = False
	reinspect_summary: dict[str, Any] | None = None
	initial_decision: DecisionResult | None = None
	preprocess_metadata: dict[str, Any] | None = None

	def response_dict(self) -> dict[str, Any]:
		return {
			"session_id": self.session_id,
			"frame_index": self.frame_index,
			"timestamp": self.timestamp,
			"model_mode": self.model_mode,
			"screen_detected": self.detection.screen_detected,
			"bbox": list(self.detection.bbox) if self.detection.bbox is not None else None,
			"detector_confidence": self.detection.detector_confidence,
			"scores": self.scores,
			"thresholds": self.thresholds,
			"passed_targets": self.decision.passed_targets,
			"predicted_label": self.decision.predicted_label,
			"decision_type": self.decision.decision_type,
			"ambiguous": self.decision.ambiguous,
			"unknown": self.decision.unknown,
			"reinspect_needed": self.decision.reinspect_needed,
			"reinspect_performed": self.reinspect_performed,
			"state_machine_allowed": self.state.state_machine_allowed,
			"effective_label": self.state.effective_label,
			"final_label": self.state.final_label,
			"state": self.state.as_dict(),
			"decision": self.decision.as_dict(),
			"initial_decision": self.initial_decision.as_dict() if self.initial_decision is not None else None,
			"reinspect_summary": self.reinspect_summary,
			"preprocess": self.preprocess_metadata,
			"artifact_paths": [artifact.as_dict() for artifact in self.artifacts],
		}
