"""Reusable model evaluation CLI.

This script intentionally stays model-agnostic at the storage boundary:
it reads per-sample records, computes binary classification metrics, and
writes one timestamped JSON summary plus one CSV row summary.

That split keeps data collection and scoring decoupled. The upstream model
or pipeline can emit records in any format the team prefers, as long as the
records can be mapped to ``ground_truth`` and ``predicted_label`` (or a
numeric ``score`` with a threshold).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.core.paths import OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "evaluation_runs"
DEFAULT_POSITIVE_LABEL = "yes"
DEFAULT_NEGATIVE_LABEL = "no"


@dataclass(slots=True)
class EvaluationRecord:
	sample_id: str
	ground_truth: str
	predicted_label: str
	score: float | None = None
	source_path: str | None = None
	prediction_source: str = "label"
	valid: bool = True
	reason: str | None = None


@dataclass(slots=True)
class MisclassifiedRecord:
	sample_id: str
	ground_truth: str
	predicted_label: str
	score: float | None
	source_path: str | None
	prediction_source: str


@dataclass(slots=True)
class EvaluationSummary:
	run_timestamp: str
	input_path: str
	output_dir: str
	input_format: str
	positive_label: str
	negative_label: str
	threshold: float | None
	total_records: int
	valid_records: int
	skipped_records: int
	unknown_labels: int
	accuracy: float
	precision: float
	recall: float
	specificity: float
	f1: float
	balanced_accuracy: float
	mcc: float
	positive_support: int
	negative_support: int
	tp: int
	fp: int
	tn: int
	fn: int
	average_score: float | None = None
	misclassified: list[MisclassifiedRecord] = field(default_factory=list)
	notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute evaluation metrics from per-sample prediction records")
	parser.add_argument("--input", type=Path, required=True, help="CSV, JSON, or JSONL file containing evaluation records")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--name", type=str, default="evaluation")
	parser.add_argument("--positive-label", type=str, default=DEFAULT_POSITIVE_LABEL)
	parser.add_argument("--negative-label", type=str, default=DEFAULT_NEGATIVE_LABEL)
	parser.add_argument("--threshold", type=float, default=None, help="Optional score threshold when predicted_label is missing")
	parser.add_argument("--label-column", type=str, default="ground_truth")
	parser.add_argument("--prediction-column", type=str, default="predicted_label")
	parser.add_argument("--score-column", type=str, default="score")
	parser.add_argument("--sample-id-column", type=str, default="sample_id")
	parser.add_argument("--source-path-column", type=str, default="source_path")
	parser.add_argument("--max-misclassified", type=int, default=20)
	return parser.parse_args()


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def append_summary_csv(csv_path: Path, row: dict[str, object]) -> None:
	ensure_dir(csv_path.parent)
	write_header = not csv_path.exists()
	with csv_path.open("a", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def normalize_label(value: Any) -> str:
	if value is None:
		return ""
	return str(value).strip().lower()


def coerce_float(value: Any) -> float | None:
	if value in (None, ""):
		return None
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"invalid numeric score: {value!r}") from exc


def load_csv_records(path: Path) -> list[dict[str, Any]]:
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		return [dict(row) for row in reader]


def load_json_records(path: Path) -> list[dict[str, Any]]:
	content = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(content, list):
		return [dict(item) for item in content]
	if isinstance(content, dict):
		for key in ("records", "items", "samples", "data"):
			value = content.get(key)
			if isinstance(value, list):
				return [dict(item) for item in value]
		raise ValueError(f"JSON input must be a list or contain a list field such as 'records': {path}")
	raise ValueError(f"unsupported JSON payload type: {type(content).__name__}")


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
	if not path.exists():
		raise FileNotFoundError(f"input not found: {path}")
	suffix = path.suffix.lower()
	if suffix == ".csv":
		return load_csv_records(path), "csv"
	if suffix in {".json", ".jsonl"}:
		if suffix == ".jsonl":
			records: list[dict[str, Any]] = []
			for line in path.read_text(encoding="utf-8").splitlines():
				line = line.strip()
				if not line:
					continue
				records.append(dict(json.loads(line)))
			return records, "jsonl"
		return load_json_records(path), "json"
	raise ValueError(f"unsupported input format: {path.suffix}")


def build_evaluation_record(
	row: dict[str, Any],
	*,
	label_column: str,
	prediction_column: str,
	score_column: str,
	sample_id_column: str,
	source_path_column: str,
	positive_label: str,
	negative_label: str,
	threshold: float | None,
) -> EvaluationRecord:
	ground_truth = normalize_label(row.get(label_column))
	if not ground_truth:
		raise ValueError(f"missing ground-truth label column: {label_column}")

	raw_prediction = row.get(prediction_column)
	predicted_label = normalize_label(raw_prediction)
	score = coerce_float(row.get(score_column))
	prediction_source = "label"
	if not predicted_label:
		if score is None:
			raise ValueError(f"missing prediction column {prediction_column!r} and score column {score_column!r}")
		if threshold is None:
			raise ValueError("threshold is required when predicted_label is missing and score is used")
		predicted_label = positive_label if score >= threshold else negative_label
		prediction_source = "score_threshold"

	sample_id = str(row.get(sample_id_column) or row.get(source_path_column) or row.get("id") or "")
	source_path = row.get(source_path_column)
	source_path_str = None if source_path in (None, "") else str(source_path)
	valid = ground_truth in {positive_label, negative_label} and predicted_label in {positive_label, negative_label}
	reason = None
	if not valid:
		reason = "label_outside_binary_scope"

	return EvaluationRecord(
		sample_id=sample_id,
		ground_truth=ground_truth,
		predicted_label=predicted_label,
		score=score,
		source_path=source_path_str,
		prediction_source=prediction_source,
		valid=valid,
		reason=reason,
	)


def confusion_counts(records: Iterable[EvaluationRecord], positive_label: str, negative_label: str) -> tuple[int, int, int, int, int, int, list[EvaluationRecord]]:
	tp = fp = tn = fn = 0
	positive_support = 0
	negative_support = 0
	valid_records: list[EvaluationRecord] = []
	for record in records:
		if not record.valid:
			continue
		valid_records.append(record)
		actual_positive = record.ground_truth == positive_label
		predicted_positive = record.predicted_label == positive_label
		if actual_positive:
			positive_support += 1
		else:
			negative_support += 1
		if actual_positive and predicted_positive:
			tp += 1
		elif actual_positive and not predicted_positive:
			fn += 1
		elif not actual_positive and predicted_positive:
			fp += 1
		else:
			tn += 1
	return tp, fp, tn, fn, positive_support, negative_support, valid_records


def safe_div(numerator: float, denominator: float) -> float:
	return numerator / denominator if denominator else 0.0


def matthews_correlation_coefficient(tp: int, fp: int, tn: int, fn: int) -> float:
	numerator = (tp * tn) - (fp * fn)
	denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	return safe_div(float(numerator), float(denominator))


def summarize_records(
	records: list[EvaluationRecord],
	*,
	input_path: Path,
	output_dir: Path,
	input_format: str,
	positive_label: str,
	negative_label: str,
	threshold: float | None,
	max_misclassified: int,
) -> EvaluationSummary:
	tp, fp, tn, fn, positive_support, negative_support, valid_records = confusion_counts(records, positive_label, negative_label)
	valid_count = len(valid_records)
	if valid_count == 0:
		raise ValueError("no valid binary records were found")

	accuracy = safe_div(float(tp + tn), float(valid_count))
	precision = safe_div(float(tp), float(tp + fp))
	recall = safe_div(float(tp), float(tp + fn))
	specificity = safe_div(float(tn), float(tn + fp))
	f1 = safe_div(2.0 * precision * recall, precision + recall)
	balanced_accuracy = (recall + specificity) / 2.0
	mcc = matthews_correlation_coefficient(tp, fp, tn, fn)
	average_score = None
	scores = [record.score for record in valid_records if record.score is not None]
	if scores:
		average_score = float(sum(scores) / len(scores))

	misclassified_records = [
		MisclassifiedRecord(
			sample_id=record.sample_id,
			ground_truth=record.ground_truth,
			predicted_label=record.predicted_label,
			score=record.score,
			source_path=record.source_path,
			prediction_source=record.prediction_source,
		)
		for record in valid_records
		if record.ground_truth != record.predicted_label
	]
	misclassified_records = misclassified_records[:max(0, max_misclassified)]
	unknown_labels = sum(1 for record in records if not record.valid)
	skipped_records = len(records) - valid_count

	notes: list[str] = []
	if threshold is None:
		notes.append("predicted_label 컬럼을 사용했기 때문에 threshold는 적용하지 않았습니다.")
	else:
		notes.append("score가 비어 있는 행은 predicted_label 컬럼을 그대로 사용했습니다.")
	if unknown_labels:
		notes.append(f"binary scope 밖의 레코드 {unknown_labels}개는 지표 계산에서 제외했습니다.")

	return EvaluationSummary(
		run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
		input_path=str(input_path),
		output_dir=str(output_dir),
		input_format=input_format,
		positive_label=positive_label,
		negative_label=negative_label,
		threshold=threshold,
		total_records=len(records),
		valid_records=valid_count,
		skipped_records=skipped_records,
		unknown_labels=unknown_labels,
		accuracy=accuracy,
		precision=precision,
		recall=recall,
		specificity=specificity,
		f1=f1,
		balanced_accuracy=balanced_accuracy,
		mcc=mcc,
		positive_support=positive_support,
		negative_support=negative_support,
		tp=tp,
		fp=fp,
		tn=tn,
		fn=fn,
		average_score=average_score,
		misclassified=misclassified_records,
		notes=notes,
	)


def evaluate(args: argparse.Namespace) -> EvaluationSummary:
	records_raw, input_format = load_records(args.input)
	records = [
		build_evaluation_record(
			row,
			label_column=args.label_column,
			prediction_column=args.prediction_column,
			score_column=args.score_column,
			sample_id_column=args.sample_id_column,
			source_path_column=args.source_path_column,
			positive_label=normalize_label(args.positive_label),
			negative_label=normalize_label(args.negative_label),
			threshold=args.threshold,
		)
		for row in records_raw
	]

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"{args.name}_{run_ts}"
	run_dir = args.output_dir / run_name
	ensure_dir(run_dir)

	summary = summarize_records(
		records,
		input_path=args.input,
		output_dir=run_dir,
		input_format=input_format,
		positive_label=normalize_label(args.positive_label),
		negative_label=normalize_label(args.negative_label),
		threshold=args.threshold,
		max_misclassified=args.max_misclassified,
	)
	summary.run_timestamp = run_ts
	summary.output_dir = str(run_dir)
	(run_dir / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
	append_summary_csv(run_dir / "run_summary.csv", {
		"run_timestamp": summary.run_timestamp,
		"input_path": summary.input_path,
		"input_format": summary.input_format,
		"output_dir": summary.output_dir,
		"positive_label": summary.positive_label,
		"negative_label": summary.negative_label,
		"threshold": summary.threshold,
		"total_records": summary.total_records,
		"valid_records": summary.valid_records,
		"skipped_records": summary.skipped_records,
		"unknown_labels": summary.unknown_labels,
		"accuracy": summary.accuracy,
		"precision": summary.precision,
		"recall": summary.recall,
		"specificity": summary.specificity,
		"f1": summary.f1,
		"balanced_accuracy": summary.balanced_accuracy,
		"mcc": summary.mcc,
		"positive_support": summary.positive_support,
		"negative_support": summary.negative_support,
		"tp": summary.tp,
		"fp": summary.fp,
		"tn": summary.tn,
		"fn": summary.fn,
		"average_score": summary.average_score,
	})
	return summary


def main() -> None:
	args = parse_args()
	summary = evaluate(args)
	print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
