from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = BASE_DIR / "outputs" / "yolo_runs" / "screen_train-2" / "weights" / "best.pt"
DEFAULT_SOURCES = [BASE_DIR / "images" / "sample_images", BASE_DIR / "images" / "sample_images_from_videos"]
DEFAULT_PROJECT = BASE_DIR / "outputs" / "yolo_test_runs"


@dataclass(slots=True)
class PredictionSummary:
	run_timestamp: str
	weights: str
	sources: list[str]
	output_dir: str
	images_seen: int
	images_with_boxes: int
	total_boxes: int
	confidence_threshold: float
	image_size: int
	device: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model")
	parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument("--source", type=Path, nargs="*", default=DEFAULT_SOURCES)
	parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
	parser.add_argument("--name", type=str, default="screen_test")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--save-txt", action="store_true")
	parser.add_argument("--save-crop", action="store_true")
	parser.add_argument("--save-conf", action="store_true")
	return parser.parse_args()


def predict_and_save(args: argparse.Namespace) -> PredictionSummary:
	if not args.weights.exists():
		raise FileNotFoundError(f"weights not found: {args.weights}")
	if not args.source:
		raise ValueError("at least one source path is required")
	for source in args.source:
		if not source.exists():
			raise FileNotFoundError(f"source not found: {source}")

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"{args.name}_{run_ts}"
	output_dir = args.project / run_name
	output_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(args.weights))
	results = []
	for source in args.source:
		results.extend(
			model.predict(
				source=str(source),
				conf=args.conf,
				imgsz=args.imgsz,
				device=args.device,
				save=True,
				save_txt=args.save_txt,
				save_crop=args.save_crop,
				save_conf=args.save_conf,
				project=str(args.project),
				name=run_name,
				exist_ok=True,
			)
		)

	images_seen = len(results)
	images_with_boxes = 0
	total_boxes = 0
	for result in results:
		boxes = getattr(result, "boxes", None)
		box_count = len(boxes) if boxes is not None else 0
		total_boxes += box_count
		if box_count > 0:
			images_with_boxes += 1

	summary = PredictionSummary(
		run_timestamp=run_ts,
		weights=str(args.weights),
		sources=[str(source) for source in args.source],
		output_dir=str(output_dir),
		images_seen=images_seen,
		images_with_boxes=images_with_boxes,
		total_boxes=total_boxes,
		confidence_threshold=args.conf,
		image_size=args.imgsz,
		device=args.device,
	)
	(output_dir / "run_summary.json").write_text(json.dumps(summary.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
	return summary


def main() -> None:
	args = parse_args()
	summary = predict_and_save(args)
	print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
