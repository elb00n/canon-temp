"""Operational inference orchestration with switchable real/mock adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import cv2
import numpy as np

from app.core.config import SETTINGS
from app.core.paths import ASSET_WEIGHTS_DIR, OUTPUTS_DIR, yolo_weight_file
from app.service.classifier_adapter import MockTargetClassifierSuite, RealTargetClassifierSuite
from app.service.decision_engine import DecisionEngine, average_scores
from app.service.detector_adapter import MockScreenDetector, ScreenDetectorAdapter, YoloDetectorAdapter
from app.service.operational_db import DEFAULT_OPERATIONAL_DB_PATH, OperationalLogStore
from app.service.operational_types import (
	ArtifactRecord,
	ClassifierResult,
	DecisionResult,
	DetectionResult,
	FrameInferenceResult,
	PreprocessResult,
	StateTransition,
	TARGET_LABELS,
	canonical_target_name,
)
from app.service.preprocess_service import ScreenPreprocessor, default_preprocess_variant, reinspect_preprocess_variants
from app.service.state_service import SequenceStateRegistry


OPERATIONAL_RUNS_DIR = OUTPUTS_DIR / "operational_runs"
ModelMode = Literal["mock", "real"]


@dataclass(slots=True)
class OperationalPipelineConfig:
	model_mode: ModelMode = "real"
	output_root: Path = OPERATIONAL_RUNS_DIR
	db_path: Path = DEFAULT_OPERATIONAL_DB_PATH
	save_artifacts: bool = True
	target_order: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
	reinspect_window: int = SETTINGS.operational.REINSPECT_WINDOW
	strict_sequence: bool = False
	yolo_weights: Path = yolo_weight_file()
	target_root: Path = ASSET_WEIGHTS_DIR
	device: str = SETTINGS.sequence.device
	prefer_openvino: bool = True
	yolo_conf: float = 0.25
	yolo_imgsz: int = SETTINGS.sequence.input_size
	yolo_padding_ratio: float = SETTINGS.operational.PREPROCESS_PADDING_RATIO


@dataclass(slots=True)
class FrameCoreResult:
	detection: DetectionResult
	preprocess_result: PreprocessResult | None
	classifier_result: ClassifierResult
	decision: DecisionResult
	preprocess_metadata: dict[str, object] | None


def utc_timestamp() -> str:
	return datetime.now(timezone.utc).isoformat()


def _safe_path_part(value: str) -> str:
	return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:96]


def build_synthetic_frame(
	*,
	scenario: str = "normal_target2_accept",
	width: int = 640,
	height: int = 480,
	frame_index: int = 0,
) -> np.ndarray:
	"""Create a deterministic image that exercises crop/deskew/resize."""
	y_gradient = np.linspace(28, 86, height, dtype=np.uint8).reshape(height, 1)
	x_gradient = np.linspace(20, 72, width, dtype=np.uint8).reshape(1, width)
	frame = np.zeros((height, width, 3), dtype=np.uint8)
	frame[:, :, 0] = y_gradient
	frame[:, :, 1] = x_gradient
	frame[:, :, 2] = np.uint8(42 + (frame_index % 4) * 8)

	box_width = int(width * 0.72)
	box_height = int(height * 0.62)
	x1 = (width - box_width) // 2
	y1 = (height - box_height) // 2
	x2 = x1 + box_width
	y2 = y1 + box_height
	cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 235, 225), -1)
	cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 38, 70), 4)
	cv2.line(frame, (x1 + 18, y1 + 40), (x2 - 18, y1 + 40), (80, 110, 190), 2)
	cv2.putText(
		frame,
		scenario[:26],
		(x1 + 22, y1 + 92),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.62,
		(35, 35, 35),
		2,
		cv2.LINE_AA,
	)
	return frame


class OperationalInferenceService:
	"""Run detector -> preprocess -> classifiers -> decision -> state -> SQLite."""

	def __init__(
		self,
		config: OperationalPipelineConfig | None = None,
		*,
		detector: ScreenDetectorAdapter | None = None,
		classifiers: object | None = None,
	) -> None:
		self.config = config or OperationalPipelineConfig()
		self.detector = detector or self._build_detector()
		self.preprocessor = ScreenPreprocessor()
		self.classifiers = classifiers or self._build_classifiers()
		self.decision_engine = DecisionEngine()
		self.store = OperationalLogStore(self.config.db_path)
		self.states = SequenceStateRegistry()

	def _build_detector(self) -> ScreenDetectorAdapter:
		if self.config.model_mode == "mock":
			return MockScreenDetector()
		return YoloDetectorAdapter(
			self.config.yolo_weights,
			device=self.config.device,
			conf=self.config.yolo_conf,
			imgsz=self.config.yolo_imgsz,
			padding_ratio=self.config.yolo_padding_ratio,
		)

	def _build_classifiers(self) -> object:
		if self.config.model_mode == "mock":
			return MockTargetClassifierSuite()
		return RealTargetClassifierSuite(
			target_root=self.config.target_root,
			device=self.config.device,
			prefer_openvino=self.config.prefer_openvino,
		)

	def _empty_scores(self) -> dict[str, float]:
		return {target_name: 0.0 for target_name in TARGET_LABELS}

	def model_info(self) -> dict[str, object]:
		info: dict[str, object] = {
			"model_mode": self.config.model_mode,
			"detector": getattr(self.detector, "detector_name", self.detector.__class__.__name__),
		}
		if self.config.model_mode == "real":
			info["yolo_weights"] = str(self.config.yolo_weights)
			if hasattr(self.classifiers, "backends_used"):
				info["classifier_backends"] = self.classifiers.backends_used()
			if hasattr(self.classifiers, "weights_used"):
				info["classifier_weights"] = self.classifiers.weights_used()
		return info

	def _run_model_stack(
		self,
		image_bgr: np.ndarray,
		*,
		scenario: str,
		frame_index: int,
		allow_reinspect: bool = True,
		check_detector_quality: bool = True,
	) -> FrameCoreResult:
		detection = self.detector.detect(image_bgr, scenario=scenario)
		if not detection.screen_detected or detection.bbox is None:
			classifier_result = ClassifierResult(scores=self._empty_scores(), predictions={}, scenario=scenario)
			decision = self.decision_engine.decide(
				classifier_result.scores,
				screen_detected=False,
				detector_confidence=detection.detector_confidence,
				allow_reinspect=allow_reinspect,
				check_detector_quality=check_detector_quality,
			)
			return FrameCoreResult(
				detection=detection,
				preprocess_result=None,
				classifier_result=classifier_result,
				decision=decision,
				preprocess_metadata=None,
			)

		preprocess_result = self.preprocessor.preprocess(
			image_bgr,
			detection.bbox,
			default_preprocess_variant(),
		)
		classifier_result = self.classifiers.predict_all(
			preprocess_result,
			scenario=scenario,
			frame_index=frame_index,
		)
		decision = self.decision_engine.decide(
			classifier_result.scores,
			screen_detected=True,
			detector_confidence=detection.detector_confidence,
			allow_reinspect=allow_reinspect,
			check_detector_quality=check_detector_quality,
		)
		return FrameCoreResult(
			detection=detection,
			preprocess_result=preprocess_result,
			classifier_result=classifier_result,
			decision=decision,
			preprocess_metadata=preprocess_result.metadata(),
		)

	def _run_image_reinspect(
		self,
		image_bgr: np.ndarray,
		*,
		detection: DetectionResult,
		scenario: str,
		frame_index: int,
	) -> tuple[DecisionResult, dict[str, object]]:
		if detection.bbox is None:
			return self.decision_engine.decide(
				self._empty_scores(),
				screen_detected=False,
				detector_confidence=detection.detector_confidence,
				allow_reinspect=False,
				check_detector_quality=False,
			), {"mode": "image", "variant_results": [], "aggregated_scores": self._empty_scores()}

		variant_results: list[dict[str, object]] = []
		score_sets: list[dict[str, float]] = []
		for variant in reinspect_preprocess_variants():
			preprocess_result = self.preprocessor.preprocess(image_bgr, detection.bbox, variant)
			classifier_result = self.classifiers.predict_all(
				preprocess_result,
				scenario=scenario,
				frame_index=frame_index,
			)
			score_sets.append(classifier_result.scores)
			variant_results.append(
				{
					"variant": variant.name,
					"scores": classifier_result.scores,
					"preprocess": preprocess_result.metadata(),
				}
			)

		aggregated_scores = average_scores(score_sets)
		final_decision = self.decision_engine.decide(
			aggregated_scores,
			screen_detected=True,
			detector_confidence=detection.detector_confidence,
			allow_reinspect=False,
			check_detector_quality=False,
		)
		return final_decision, {
			"mode": "image",
			"aggregation_method": "average",
			"variant_results": variant_results,
			"aggregated_scores": aggregated_scores,
		}

	def _save_artifacts(
		self,
		*,
		session_id: str,
		frame_index: int,
		image_bgr: np.ndarray,
		preprocess_result: PreprocessResult | None,
		enabled: bool,
	) -> list[ArtifactRecord]:
		if not enabled:
			return []
		frame_dir = self.config.output_root / _safe_path_part(session_id) / f"frame_{frame_index:06d}"
		frame_dir.mkdir(parents=True, exist_ok=True)
		artifacts: list[ArtifactRecord] = []

		original_path = frame_dir / "input.jpg"
		cv2.imwrite(str(original_path), image_bgr)
		artifacts.append(ArtifactRecord("input_image", original_path, {"shape": list(image_bgr.shape)}))

		if preprocess_result is None:
			return artifacts

		crop_path = frame_dir / f"{preprocess_result.variant.name}_crop.jpg"
		processed_path = frame_dir / f"{preprocess_result.variant.name}_processed.jpg"
		cv2.imwrite(str(crop_path), preprocess_result.crop_bgr)
		cv2.imwrite(str(processed_path), preprocess_result.processed_bgr)
		artifacts.append(ArtifactRecord("crop", crop_path, preprocess_result.metadata()))
		artifacts.append(ArtifactRecord("preprocessed", processed_path, preprocess_result.metadata()))
		return artifacts

	def _build_logged_result(
		self,
		*,
		session_id: str,
		frame_index: int,
		image_bgr: np.ndarray,
		core: FrameCoreResult,
		decision: DecisionResult,
		state: StateTransition,
		initial_decision: DecisionResult | None = None,
		reinspect_performed: bool = False,
		reinspect_summary: dict[str, object] | None = None,
		save_artifacts: bool | None = None,
	) -> FrameInferenceResult:
		artifacts = self._save_artifacts(
			session_id=session_id,
			frame_index=frame_index,
			image_bgr=image_bgr,
			preprocess_result=core.preprocess_result,
			enabled=self.config.save_artifacts if save_artifacts is None else save_artifacts,
		)
		result = FrameInferenceResult(
			session_id=session_id,
			frame_index=frame_index,
			timestamp=utc_timestamp(),
			model_mode=self.config.model_mode,
			detection=core.detection,
			decision=decision,
			state=state,
			scores=decision.scores,
			thresholds=decision.thresholds,
			artifacts=artifacts,
			reinspect_performed=reinspect_performed,
			reinspect_summary=reinspect_summary,
			initial_decision=initial_decision,
			preprocess_metadata=core.preprocess_metadata,
		)
		self.store.insert_frame_result(result)
		return result

	def infer_image(
		self,
		image_bgr: np.ndarray,
		*,
		scenario: str = "normal_target2_accept",
		session_id: str | None = None,
		frame_index: int = 0,
		target_order: list[str] | None = None,
		strict_sequence: bool | None = None,
		reset_state: bool = False,
		save_artifacts: bool | None = None,
	) -> FrameInferenceResult:
		session_id = session_id or f"session_{uuid4().hex}"
		canonical_order = [canonical_target_name(target) for target in (target_order or self.config.target_order)]
		self.store.upsert_session(
			session_id,
			mode="image",
			scenario=scenario,
			metadata={"target_order": canonical_order, **self.model_info()},
		)
		state_machine = self.states.get(
			session_id,
			target_order=canonical_order,
			strict_order=self.config.strict_sequence if strict_sequence is None else strict_sequence,
			reset=reset_state,
		)

		core = self._run_model_stack(image_bgr, scenario=scenario, frame_index=frame_index)
		initial_decision = core.decision
		decision = initial_decision
		reinspect_performed = False
		reinspect_summary = None
		if initial_decision.reinspect_needed:
			decision, reinspect_summary = self._run_image_reinspect(
				image_bgr,
				detection=core.detection,
				scenario=scenario,
				frame_index=frame_index,
			)
			reinspect_performed = True

		state = state_machine.apply(decision)
		return self._build_logged_result(
			session_id=session_id,
			frame_index=frame_index,
			image_bgr=image_bgr,
			core=core,
			decision=decision,
			state=state,
			initial_decision=initial_decision if reinspect_performed else None,
			reinspect_performed=reinspect_performed,
			reinspect_summary=reinspect_summary,
			save_artifacts=save_artifacts,
		)

	def _auxiliary_state(self, session_id: str, frame_index: int, decision: DecisionResult) -> StateTransition:
		return StateTransition(
			session_id=session_id,
			expected_label=None,
			effective_label="unknown",
			final_label="unknown",
			state_machine_allowed=False,
			current_index=frame_index,
			completed_labels=[],
			event_type="reinspect_candidate",
			reason="frame was consumed as sequence reinspect evidence",
		)

	def infer_sequence(
		self,
		frames_bgr: list[np.ndarray],
		*,
		scenario: str = "normal_target2_accept",
		session_id: str | None = None,
		target_order: list[str] | None = None,
		strict_sequence: bool | None = None,
		reset_state: bool = True,
		save_artifacts: bool | None = None,
	) -> dict[str, object]:
		if not frames_bgr:
			raise ValueError("frames_bgr must contain at least one frame")
		session_id = session_id or f"sequence_{uuid4().hex}"
		canonical_order = [canonical_target_name(target) for target in (target_order or self.config.target_order)]
		self.store.upsert_session(
			session_id,
			mode="sequence",
			scenario=scenario,
			metadata={"target_order": canonical_order, "frame_count": len(frames_bgr), **self.model_info()},
		)
		state_machine = self.states.get(
			session_id,
			target_order=canonical_order,
			strict_order=self.config.strict_sequence if strict_sequence is None else strict_sequence,
			reset=reset_state,
		)

		results: list[FrameInferenceResult] = []
		frame_index = 0
		while frame_index < len(frames_bgr):
			frame_bgr = frames_bgr[frame_index]
			core = self._run_model_stack(frame_bgr, scenario=scenario, frame_index=frame_index)
			initial_decision = core.decision
			decision = initial_decision
			reinspect_performed = False
			reinspect_summary = None
			consumed_until = frame_index

			if initial_decision.reinspect_needed:
				score_sets: list[dict[str, float]] = []
				candidate_results: list[dict[str, object]] = []
				next_index = frame_index + 1
				while next_index < len(frames_bgr) and len(score_sets) < self.config.reinspect_window:
					next_core = self._run_model_stack(
						frames_bgr[next_index],
						scenario=scenario,
						frame_index=next_index,
						allow_reinspect=False,
						check_detector_quality=False,
					)
					next_decision = next_core.decision
					if next_core.detection.screen_detected and next_core.detection.bbox is not None:
						score_sets.append(next_decision.scores)
						candidate_results.append(
							{
								"frame_index": next_index,
								"scores": next_decision.scores,
								"detector_confidence": next_core.detection.detector_confidence,
							}
						)
					aux_state = self._auxiliary_state(session_id, next_index, next_decision)
					results.append(
						self._build_logged_result(
							session_id=session_id,
							frame_index=next_index,
							image_bgr=frames_bgr[next_index],
							core=next_core,
							decision=next_decision,
							state=aux_state,
							save_artifacts=save_artifacts,
						)
					)
					consumed_until = next_index
					next_index += 1

				if not score_sets:
					score_sets.append(initial_decision.scores)
				aggregated_scores = average_scores(score_sets)
				decision = self.decision_engine.decide(
					aggregated_scores,
					screen_detected=core.detection.screen_detected,
					detector_confidence=core.detection.detector_confidence,
					allow_reinspect=False,
					check_detector_quality=False,
				)
				reinspect_performed = True
				reinspect_summary = {
					"mode": "sequence",
					"aggregation_method": "average",
					"reinspect_window": self.config.reinspect_window,
					"candidate_results": candidate_results,
					"aggregated_scores": aggregated_scores,
				}

			state = state_machine.apply(decision)
			results.append(
				self._build_logged_result(
					session_id=session_id,
					frame_index=frame_index,
					image_bgr=frame_bgr,
					core=core,
					decision=decision,
					state=state,
					initial_decision=initial_decision if reinspect_performed else None,
					reinspect_performed=reinspect_performed,
					reinspect_summary=reinspect_summary,
					save_artifacts=save_artifacts,
				)
			)
			frame_index = max(consumed_until + 1, frame_index + 1)

		ordered_results = sorted(results, key=lambda item: item.frame_index)
		final_result = ordered_results[-1].response_dict()
		return {
			"session_id": session_id,
			"mode": "sequence",
			"model_mode": self.config.model_mode,
			"scenario": scenario,
			"frame_count": len(frames_bgr),
			"results": [result.response_dict() for result in ordered_results],
			"final_result": final_result,
			"db_path": str(self.config.db_path),
		}

	def run_smoke_scenario(
		self,
		scenario: str,
		*,
		width: int = 640,
		height: int = 480,
		mode: str = "image",
		frame_count: int = 5,
	) -> dict[str, object]:
		if mode == "sequence":
			frames = [
				build_synthetic_frame(scenario=scenario, width=width, height=height, frame_index=index)
				for index in range(frame_count)
			]
			return self.infer_sequence(
				frames,
				scenario=scenario,
				target_order=["Target2", "Target3", "Target4"],
				strict_sequence=False,
			)
		frame = build_synthetic_frame(scenario=scenario, width=width, height=height)
		result = self.infer_image(
			frame,
			scenario=scenario,
			target_order=["Target2", "Target3", "Target4"],
			strict_sequence=False,
		)
		response = result.response_dict()
		response["db_path"] = str(self.config.db_path)
		return response


__all__ = [
	"OPERATIONAL_RUNS_DIR",
	"OperationalInferenceService",
	"OperationalPipelineConfig",
	"build_synthetic_frame",
]
