"""CLI wrapper for sequential target detection on live camera or stream sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.paths import ASSET_WEIGHTS_DIR, SEQUENCE_STREAM_RUNS_DIR, yolo_weight_file
from app.service import SequenceStreamService, StreamRunConfig


DEFAULT_TARGET_ORDER = ["target_1", "target_2", "target_3", "target_4"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run sequential detection on a live camera or RTSP stream")
	parser.add_argument("--source", type=str, default="0", help="Camera index, RTSP URL, or video source string")
	parser.add_argument("--source-label", type=str, default="stream", help="Label used for output folders")
	parser.add_argument("--target-order", nargs="*", default=DEFAULT_TARGET_ORDER)
	parser.add_argument("--target-root", type=str, default=str(ASSET_WEIGHTS_DIR))
	parser.add_argument("--yolo-weights", type=str, default=str(yolo_weight_file()))
	parser.add_argument("--output-root", type=str, default=str(SEQUENCE_STREAM_RUNS_DIR))
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--threshold", type=float, default=None, help="Override all target thresholds")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--padding-ratio", type=float, default=0.02)
	parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth captured frame")
	parser.add_argument("--sample-seconds", type=float, default=0.5, help="Minimum seconds between processed frames")
	parser.add_argument("--min-consecutive", type=int, default=3, help="Consecutive yes frames required to confirm a target")
	parser.add_argument("--max-missed", type=int, default=4, help="Allowed missed detections before resetting the yes streak")
	parser.add_argument("--save-confirmed-frames", action="store_true", help="Save only frames that confirm a target")
	parser.add_argument("--confirmed-pre-roll", type=int, default=3, help="How many processed frames before confirmation to save")
	parser.add_argument("--show-preview", action="store_true", help="Display a live preview window")
	parser.add_argument("--no-show-preview", action="store_false", dest="show_preview", help="Disable the preview window")
	parser.set_defaults(show_preview=True)
	parser.add_argument("--window-name", type=str, default="Sequence Stream")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = StreamRunConfig(
		source=args.source,
		source_label=args.source_label,
		target_order=args.target_order,
		target_root=Path(args.target_root),
		yolo_weights=Path(args.yolo_weights),
		output_root=Path(args.output_root),
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
		show_preview=args.show_preview,
		window_name=args.window_name,
	)
	service = SequenceStreamService(config)
	result = service.process_stream()
	print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()