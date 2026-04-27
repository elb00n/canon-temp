"""State and sequence handling after the decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.service.operational_types import DecisionResult, StateTransition, TARGET_LABELS, canonical_target_name


@dataclass(slots=True)
class SequenceStateMachine:
	session_id: str
	target_order: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
	strict_order: bool = False
	current_index: int = 0
	completed_labels: list[str] = field(default_factory=list)

	def expected_label(self) -> str | None:
		if self.current_index >= len(self.target_order):
			return None
		return self.target_order[self.current_index]

	def apply(self, decision: DecisionResult) -> StateTransition:
		expected = self.expected_label()
		if not decision.accepted:
			return StateTransition(
				session_id=self.session_id,
				expected_label=expected,
				effective_label="unknown",
				final_label="unknown",
				state_machine_allowed=False,
				current_index=self.current_index,
				completed_labels=list(self.completed_labels),
				event_type="ignored",
				reason="decision was not accepted",
			)

		effective_label = canonical_target_name(decision.predicted_label)
		if not self.strict_order:
			if effective_label in self.target_order and effective_label not in self.completed_labels:
				self.completed_labels.append(effective_label)
				self.current_index = min(len(self.target_order), max(self.current_index, self.target_order.index(effective_label) + 1))
			return StateTransition(
				session_id=self.session_id,
				expected_label=expected,
				effective_label=effective_label,
				final_label=effective_label,
				state_machine_allowed=True,
				current_index=self.current_index,
				completed_labels=list(self.completed_labels),
				event_type="accepted",
				reason="accepted by non-strict sequence state",
			)

		if expected is None:
			return StateTransition(
				session_id=self.session_id,
				expected_label=None,
				effective_label=effective_label,
				final_label="sequence_complete",
				state_machine_allowed=False,
				current_index=self.current_index,
				completed_labels=list(self.completed_labels),
				event_type="sequence_complete",
				reason="all configured sequence targets are already complete",
			)

		if effective_label != expected:
			return StateTransition(
				session_id=self.session_id,
				expected_label=expected,
				effective_label=effective_label,
				final_label="sequence_blocked",
				state_machine_allowed=False,
				current_index=self.current_index,
				completed_labels=list(self.completed_labels),
				event_type="blocked",
				reason=f"expected {expected} before {effective_label}",
			)

		self.completed_labels.append(effective_label)
		self.current_index += 1
		return StateTransition(
			session_id=self.session_id,
			expected_label=expected,
			effective_label=effective_label,
			final_label=effective_label,
			state_machine_allowed=True,
			current_index=self.current_index,
			completed_labels=list(self.completed_labels),
			event_type="accepted",
			reason="accepted by strict sequence order",
		)


class SequenceStateRegistry:
	"""In-memory state holder for FastAPI/session smoke runs."""

	def __init__(self) -> None:
		self._states: dict[str, SequenceStateMachine] = {}

	def get(
		self,
		session_id: str,
		*,
		target_order: list[str] | None = None,
		strict_order: bool = False,
		reset: bool = False,
	) -> SequenceStateMachine:
		canonical_order = [canonical_target_name(target_name) for target_name in (target_order or list(TARGET_LABELS))]
		if reset or session_id not in self._states:
			self._states[session_id] = SequenceStateMachine(
				session_id=session_id,
				target_order=canonical_order,
				strict_order=strict_order,
			)
		else:
			state = self._states[session_id]
			state.strict_order = strict_order
			state.target_order = canonical_order
		return self._states[session_id]

	def reset(self, session_id: str) -> None:
		self._states.pop(session_id, None)


__all__ = [
	"SequenceStateMachine",
	"SequenceStateRegistry",
]
