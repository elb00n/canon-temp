"""Operational decision logic for target scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import SETTINGS
from app.service.operational_types import DecisionResult, TARGET_LABELS


@dataclass(frozen=True, slots=True)
class DecisionConfig:
	thresholds: dict[str, float] = field(default_factory=lambda: dict(SETTINGS.operational.thresholds))
	decision_margin: float = SETTINGS.operational.DECISION_MARGIN
	delta_accept: float = SETTINGS.operational.DELTA_ACCEPT
	detector_confidence_threshold: float = SETTINGS.operational.DETECTOR_CONFIDENCE_THRESHOLD


class DecisionEngine:
	"""Apply real operational logic to target scores and detector quality."""

	def __init__(self, config: DecisionConfig | None = None) -> None:
		self.config = config or DecisionConfig()

	def decide(
		self,
		scores: dict[str, float],
		*,
		screen_detected: bool,
		detector_confidence: float,
		allow_reinspect: bool = True,
		check_detector_quality: bool = True,
	) -> DecisionResult:
		thresholds = dict(self.config.thresholds)
		ordered_scores = sorted(
			((target, float(scores.get(target, 0.0))) for target in TARGET_LABELS),
			key=lambda item: item[1],
			reverse=True,
		)
		top1_label, top1_score = ordered_scores[0] if ordered_scores else (None, None)
		top2_label, top2_score = ordered_scores[1] if len(ordered_scores) > 1 else (None, None)
		margin = None if top1_score is None or top2_score is None else round(top1_score - top2_score, 4)

		if not screen_detected:
			return DecisionResult(
				scores=dict(scores),
				thresholds=thresholds,
				passed_targets=[],
				predicted_label="unknown",
				decision_type="no_screen",
				ambiguous=False,
				unknown=True,
				reinspect_needed=False,
				top1_label=top1_label,
				top1_score=top1_score,
				top2_label=top2_label,
				top2_score=top2_score,
				margin=margin,
				reason="screen was not detected",
			)

		passed_targets = [
			target_name
			for target_name in TARGET_LABELS
			if float(scores.get(target_name, 0.0)) >= thresholds[target_name]
		]

		if not passed_targets:
			return DecisionResult(
				scores=dict(scores),
				thresholds=thresholds,
				passed_targets=[],
				predicted_label="unknown",
				decision_type="unknown_no_pass",
				ambiguous=False,
				unknown=True,
				reinspect_needed=False,
				top1_label=top1_label,
				top1_score=top1_score,
				top2_label=top2_label,
				top2_score=top2_score,
				margin=margin,
				reason="no target score passed its threshold",
			)

		if (
			check_detector_quality
			and detector_confidence < self.config.detector_confidence_threshold
			and allow_reinspect
		):
			return DecisionResult(
				scores=dict(scores),
				thresholds=thresholds,
				passed_targets=passed_targets,
				predicted_label=top1_label or "unknown",
				decision_type="poor_detector_reinspect",
				ambiguous=False,
				unknown=False,
				reinspect_needed=True,
				top1_label=top1_label,
				top1_score=top1_score,
				top2_label=top2_label,
				top2_score=top2_score,
				margin=margin,
				reason="detector confidence is below operational quality threshold",
			)

		if len(passed_targets) == 1:
			target_name = passed_targets[0]
			score = float(scores[target_name])
			required_score = thresholds[target_name] + self.config.delta_accept
			if score >= required_score:
				return DecisionResult(
					scores=dict(scores),
					thresholds=thresholds,
					passed_targets=passed_targets,
					predicted_label=target_name,
					decision_type="single_pass_accept",
					ambiguous=False,
					unknown=False,
					reinspect_needed=False,
					top1_label=top1_label,
					top1_score=top1_score,
					top2_label=top2_label,
					top2_score=top2_score,
					margin=margin,
					reason="one target passed and exceeded the delta accept band",
				)
			return DecisionResult(
				scores=dict(scores),
				thresholds=thresholds,
				passed_targets=passed_targets,
				predicted_label=target_name,
				decision_type="borderline_single_pass",
				ambiguous=False,
				unknown=False,
				reinspect_needed=True,
				top1_label=top1_label,
				top1_score=top1_score,
				top2_label=top2_label,
				top2_score=top2_score,
				margin=margin,
				reason="single passing target did not exceed threshold by DELTA_ACCEPT",
			)

		top1_threshold = thresholds[top1_label] if top1_label is not None else 1.0
		clear_margin = margin is not None and margin >= self.config.decision_margin
		clear_delta = top1_score is not None and top1_score >= top1_threshold + self.config.delta_accept
		if clear_margin and clear_delta:
			return DecisionResult(
				scores=dict(scores),
				thresholds=thresholds,
				passed_targets=passed_targets,
				predicted_label=top1_label or "unknown",
				decision_type="multi_pass_clear_winner",
				ambiguous=False,
				unknown=False,
				reinspect_needed=False,
				top1_label=top1_label,
				top1_score=top1_score,
				top2_label=top2_label,
				top2_score=top2_score,
				margin=margin,
				reason="multiple targets passed, but top1/top2 margin is clear",
			)

		return DecisionResult(
			scores=dict(scores),
			thresholds=thresholds,
			passed_targets=passed_targets,
			predicted_label=top1_label or "unknown",
			decision_type="ambiguous",
			ambiguous=True,
			unknown=False,
			reinspect_needed=True,
			top1_label=top1_label,
			top1_score=top1_score,
			top2_label=top2_label,
			top2_score=top2_score,
			margin=margin,
			reason="multiple targets passed without enough top1/top2 margin",
		)


def average_scores(score_sets: list[dict[str, float]]) -> dict[str, float]:
	if not score_sets:
		return {target_name: 0.0 for target_name in TARGET_LABELS}
	return {
		target_name: round(
			sum(float(scores.get(target_name, 0.0)) for scores in score_sets) / len(score_sets),
			4,
		)
		for target_name in TARGET_LABELS
	}


__all__ = [
	"DecisionConfig",
	"DecisionEngine",
	"average_scores",
]
