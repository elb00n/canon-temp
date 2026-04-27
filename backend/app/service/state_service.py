"""State and sequence handling after the decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.service.operational_types import DecisionResult, StateTransition, TARGET_LABELS, canonical_target_name


REQUIRED_CONSECUTIVE_FRAMES = 3


@dataclass(slots=True)
class SequenceStateMachine:
	session_id: str
	target_order: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
	strict_order: bool = False
	current_index: int = 0
	completed_labels: list[str] = field(default_factory=list)
	required_consecutive: int = REQUIRED_CONSECUTIVE_FRAMES
	consecutive_count: int = 0
	restart_count: int = 0

	def expected_label(self) -> str | None:
		if self.current_index >= len(self.target_order):
			return None
		return self.target_order[self.current_index]

	def apply(self, decision: DecisionResult) -> StateTransition:
		expected = self.expected_label()
		if not decision.accepted:
			self.consecutive_count = 0
			self.restart_count = 0
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
			first_target = self.target_order[0] if self.target_order else None
			if first_target is not None and effective_label == first_target:
				self.restart_count += 1
				if self.restart_count < self.required_consecutive:
					return StateTransition(
						session_id=self.session_id,
						expected_label=None,
						effective_label=effective_label,
						final_label="sequence_complete",
						state_machine_allowed=False,
						current_index=self.current_index,
						completed_labels=list(self.completed_labels),
						event_type="restart_pending",
						reason=f"restart pending {self.restart_count}/{self.required_consecutive}",
					)
				self.current_index = 0
				self.completed_labels = []
				self.consecutive_count = 0
				self.restart_count = 0
				expected = self.expected_label()
			else:
				self.restart_count = 0
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
			self.consecutive_count = 0
			if effective_label in self.completed_labels:
				return StateTransition(
					session_id=self.session_id,
					expected_label=expected,
					effective_label=effective_label,
					final_label=effective_label,
					state_machine_allowed=True,
					current_index=self.current_index,
					completed_labels=list(self.completed_labels),
					event_type="awaiting_next",
					reason=f"{effective_label} already confirmed; awaiting {expected}",
				)
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

		self.consecutive_count += 1
		if self.consecutive_count < self.required_consecutive:
			return StateTransition(
				session_id=self.session_id,
				expected_label=expected,
				effective_label=effective_label,
				final_label=effective_label,
				state_machine_allowed=True,
				current_index=self.current_index,
				completed_labels=list(self.completed_labels),
				event_type="counting",
				reason=f"{self.consecutive_count}/{self.required_consecutive}",
			)

		self.completed_labels.append(effective_label)
		self.current_index += 1
		self.consecutive_count = 0
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
