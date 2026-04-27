"""Map operational pipeline results to frontend transport payloads."""

from __future__ import annotations

import json
from typing import Any


def _response_dict(result: Any) -> dict[str, Any]:
	if hasattr(result, "response_dict"):
		return result.response_dict()
	if isinstance(result, dict):
		return dict(result)
	raise TypeError(f"unsupported operational result type: {type(result)!r}")


def _top_confidence(response: dict[str, Any]) -> float:
	decision = response.get("decision")
	if isinstance(decision, dict) and decision.get("top1_score") is not None:
		return float(decision["top1_score"])

	scores = response.get("scores")
	if isinstance(scores, dict) and scores:
		return float(max(float(value) for value in scores.values()))
	return 0.0


def _current_step_index(response: dict[str, Any]) -> int:
	state = response.get("state")
	current_index = 0
	if isinstance(state, dict):
		current_index = int(state.get("current_index") or 0)
	return max(1, min(4, current_index or 1))


def _status_label(response: dict[str, Any]) -> str:
	if bool(response.get("state_machine_allowed")):
		return "Success"
	if bool(response.get("ambiguous")):
		return "Ambiguous"
	if bool(response.get("reinspect_needed")) or bool(response.get("reinspect_performed")):
		return "Reinspect"
	if bool(response.get("unknown")):
		return "Unknown"

	state = response.get("state")
	if isinstance(state, dict):
		event_type = str(state.get("event_type") or "")
		if event_type:
			return event_type.replace("_", " ").title()
	return "Analyzing"


def _system_message(response: dict[str, Any]) -> str:
	final_label = str(response.get("final_label") or "unknown")
	if bool(response.get("state_machine_allowed")):
		return f"{final_label} OK"
	if bool(response.get("ambiguous")):
		return "AMBIGUOUS"
	if bool(response.get("reinspect_needed")) or bool(response.get("reinspect_performed")):
		return "REINSPECT"
	if bool(response.get("unknown")):
		return "UNKNOWN"
	return final_label.upper()


def _first_artifact_path(response: dict[str, Any], artifact_type: str = "input_image") -> str:
	for artifact in response.get("artifact_paths") or []:
		if isinstance(artifact, dict) and artifact.get("artifact_type") == artifact_type:
			return str(artifact.get("path") or "")
	return ""


def operational_result_to_frontend_payload(
	result: Any,
	*,
	include_detail: bool = True,
) -> dict[str, Any]:
	response = _response_dict(result)
	state = response.get("state") if isinstance(response.get("state"), dict) else {}
	decision = response.get("decision") if isinstance(response.get("decision"), dict) else {}

	confidence = _top_confidence(response)
	is_unknown = bool(response.get("unknown"))
	ambiguous = bool(response.get("ambiguous"))
	reinspect_needed = bool(response.get("reinspect_needed"))
	allowed_transition = bool(response.get("state_machine_allowed"))
	current_step_index = _current_step_index(response)
	confirmed_state = _status_label(response)
	system_message = _system_message(response)

	payload: dict[str, Any] = {
		"timestamp": response.get("timestamp"),
		"predicted_label": response.get("predicted_label", "unknown"),
		"final_label": response.get("final_label", "unknown"),
		"effective_label": response.get("effective_label", "unknown"),
		"confidence": confidence,
		"confirmed_state": confirmed_state,
		"allowed_transition": allowed_transition,
		"inference": True,
		"is_unknown": is_unknown,
		"ambiguous": ambiguous,
		"reinspect_needed": reinspect_needed,
		"logic": {
			"predicted_label": response.get("predicted_label", "unknown"),
			"final_label": response.get("final_label", "unknown"),
			"effective_label": response.get("effective_label", "unknown"),
			"expected_label": state.get("expected_label"),
			"completed_labels": list(state.get("completed_labels") or []),
			"confirmed_state": confirmed_state,
			"allowed_transition": allowed_transition,
			"current_step_index": current_step_index,
			"confidence": confidence,
			"is_unknown": is_unknown,
			"ambiguous": ambiguous,
			"reinspect_needed": reinspect_needed,
			"decision_type": response.get("decision_type"),
			"state_event_type": state.get("event_type"),
			"state_reason": state.get("reason"),
			"decision_reason": decision.get("reason"),
		},
		"display": {
			"system_message": system_message,
		},
	}
	if include_detail:
		payload["operational_detail"] = response
	return payload


def camera_state_message(camera_id: str, result: Any) -> dict[str, Any]:
	return {
		"cameraId": camera_id,
		"payload": operational_result_to_frontend_payload(result),
	}


def operational_response_to_log(
	result: Any,
	*,
	log_id: str | int | None = None,
	source_type: str = "operational",
	file_path: str = "",
	cam_id: str = "",
) -> dict[str, Any]:
	response = _response_dict(result)
	payload = operational_result_to_frontend_payload(response, include_detail=False)
	file_path = file_path or _first_artifact_path(response)
	anomaly_flag = (
		bool(payload.get("is_unknown"))
		or bool(payload.get("ambiguous"))
		or bool(payload.get("reinspect_needed"))
		or not bool(payload.get("allowed_transition"))
	)
	record: dict[str, Any] = {
		"id": log_id or response.get("frame_index", 0),
		"timestamp": response.get("timestamp"),
		"source_type": source_type,
		"confirmed_state": payload["confirmed_state"],
		"predicted_label": payload["predicted_label"],
		"final_label": payload["final_label"],
		"confidence": payload["confidence"],
		"anomaly_flag": anomaly_flag,
		"file_path": file_path,
		"cam_id": cam_id,
		"target_idx": payload["logic"]["current_step_index"],
		"extra_json": json.dumps(
			{
				"source": "operational_pipeline",
				"session_id": response.get("session_id"),
				"frame_index": response.get("frame_index"),
				"decision_type": response.get("decision_type"),
				"state": response.get("state"),
			},
			ensure_ascii=False,
		),
	}
	return record


__all__ = [
	"camera_state_message",
	"operational_response_to_log",
	"operational_result_to_frontend_payload",
]
