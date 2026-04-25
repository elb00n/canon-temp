"""Thin CLI wrapper for target testing on sample folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.paths import ASSET_WEIGHTS_DIR, OUTPUTS_DIR, TARGET_TEST_RUNS_DIR, yolo_weight_file
from app.service import TargetTestRunConfig, TargetTestService


DEFAULT_TARGET_MODEL_ROOT = ASSET_WEIGHTS_DIR
DEFAULT_YOLO_WEIGHTS = yolo_weight_file()
DEFAULT_SOURCES = [OUTPUTS_DIR / "warped"]
DEFAULT_OUTPUT_DIR = TARGET_TEST_RUNS_DIR


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run YOLO detection and target yes/no classification on sample folders")
	parser.add_argument("--target-name", type=str, required=True, help="Target name, e.g. target_1")
	parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_MODEL_ROOT)
	parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
	parser.add_argument("--source", type=Path, nargs="*", default=DEFAULT_SOURCES)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--threshold", type=float, default=None)
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--padding-ratio", type=float, default=0.02)
	parser.add_argument("--save-crops", action="store_true", help="Save raw crops for each detection")
	parser.add_argument("--save-contour-warped", action="store_true", help="Save contour-warped crops for inspection")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = TargetTestRunConfig(
		target_name=args.target_name,
		target_root=args.target_root,
		yolo_weights=args.yolo_weights,
		source=args.source,
		output_dir=args.output_dir,
		device=args.device,
		threshold=args.threshold,
		conf=args.conf,
		imgsz=args.imgsz,
		padding_ratio=args.padding_ratio,
		save_crops=args.save_crops,
		save_contour_warped=args.save_contour_warped,
	)
	service = TargetTestService(config)
	summary = service.run()
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()