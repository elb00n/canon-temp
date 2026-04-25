"""CLI wrapper for sequential target detection on videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.paths import ASSET_WEIGHTS_DIR, SAMPLE_VIDEO_DIR, SEQUENCE_VIDEO_RUNS_DIR, yolo_weight_file
from app.service import SequenceRunConfig, SequenceService


DEFAULT_TARGET_ORDER = ["target_1", "target_2", "target_3", "target_4"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run sequential video detection with YOLO and target classifiers")
	parser.add_argument("--source", type=Path, nargs="*", default=[SAMPLE_VIDEO_DIR])
	parser.add_argument("--target-order", nargs="*", default=DEFAULT_TARGET_ORDER)
	parser.add_argument("--target-root", type=Path, default=ASSET_WEIGHTS_DIR)
	parser.add_argument("--yolo-weights", type=Path, default=yolo_weight_file())
	parser.add_argument("--output-root", type=Path, default=SEQUENCE_VIDEO_RUNS_DIR)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--threshold", type=float, default=None, help="Override all target thresholds")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--padding-ratio", type=float, default=0.02)
	parser.add_argument("--frame-step", type=float, default=1.0, help="Process every Nth frame")
	parser.add_argument("--sample-seconds", type=float, default=0.5, help="Sample one frame every N seconds based on video FPS")
	parser.add_argument("--min-consecutive", type=int, default=3, help="Consecutive yes frames required to confirm a target")
	parser.add_argument("--max-missed", type=int, default=4, help="Allowed missed detections before resetting the yes streak")
	parser.add_argument("--save-confirmed-frames", action="store_true", help="Save only frames that confirm a target")
	parser.add_argument("--confirmed-pre-roll", type=int, default=3, help="How many processed frames before confirmation to save")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = SequenceRunConfig(
		source=args.source,
		target_order=args.target_order,
		target_root=args.target_root,
		yolo_weights=args.yolo_weights,
		output_root=args.output_root,
		device=args.device,
		threshold=args.threshold,
		conf=args.conf,
		imgsz=args.imgsz,
		padding_ratio=args.padding_ratio,
		frame_step=args.frame_step,
		sample_seconds=args.sample_seconds,
		min_consecutive=args.min_consecutive,
		max_missed=args.max_missed,
		save_confirmed_frames=args.save_confirmed_frames,
		confirmed_pre_roll=args.confirmed_pre_roll,
	)
	service = SequenceService(config)
	result = service.run()
	print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
