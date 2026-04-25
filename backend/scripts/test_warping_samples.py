"""Run YOLO detection and warping on sample image folders."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.paths import ASSET_WEIGHTS_DIR, OUTPUTS_DIR, SAMPLE_IMAGE_DIR, SAMPLE_IMAGES_FROM_VIDEOS_DIR, WARPING_TEST_RUNS_DIR, yolo_weight_file
from app.models.warping import YoloScreenWarper, list_image_files, warp_screen_from_crop


DEFAULT_WEIGHTS = yolo_weight_file()
DEFAULT_SOURCES = [SAMPLE_IMAGE_DIR, SAMPLE_IMAGES_FROM_VIDEOS_DIR]
DEFAULT_OUTPUT_DIR = WARPING_TEST_RUNS_DIR


@dataclass(slots=True)
class SampleWarpSummary:
	run_timestamp: str
	weights: str
	sources: list[str]
	output_dir: str
	images_seen: int
	preview_saved: int
	detections_seen: int
	warped_candidates_seen: int
	contour_warped_saved: int
	warped_saved: int
	device: str
	confidence_threshold: float
	image_size: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run YOLO11n warping test on sample folders")
	parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument("--source", type=Path, nargs="*", default=DEFAULT_SOURCES)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--padding-ratio", type=float, default=0.02)
	parser.add_argument("--output-size", type=int, default=640)
	parser.add_argument("--save-crops", action="store_true", help="Save raw crops as well as warped images")
	parser.add_argument("--save-contour-warped", action="store_true", help="Save contour-warped images for inspection")
	return parser.parse_args()


def draw_detection_preview(image_bgr: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
	preview = image_bgr.copy()
	for detection in detections:
		bbox_xyxy = detection["bbox_xyxy"]
		x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox_xyxy]
		label = f'{detection["class_name"]} {float(detection["confidence"]):.2f}'
		cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
		label_y = max(18, y_min - 8)
		cv2.putText(preview, label, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
	return preview


def run_sample_test(args: argparse.Namespace) -> SampleWarpSummary:
	if not args.weights.exists():
		raise FileNotFoundError(f"weights not found: {args.weights}")
	if not args.source:
		raise ValueError("at least one source path is required")
	for source in args.source:
		if not source.exists():
			raise FileNotFoundError(f"source not found: {source}")

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"warping_{run_ts}"
	output_dir = args.output_dir / run_name
	preview_dir = output_dir / "preview"
	warped_dir = output_dir / "warped"
	crop_dir = output_dir / "crops"
	contour_warped_dir = output_dir / "contour_warped"
	preview_dir.mkdir(parents=True, exist_ok=True)
	warped_dir.mkdir(parents=True, exist_ok=True)
	if args.save_crops:
		crop_dir.mkdir(parents=True, exist_ok=True)
	if args.save_contour_warped:
		contour_warped_dir.mkdir(parents=True, exist_ok=True)

	warper = YoloScreenWarper(
		weights=args.weights,
		device=args.device,
		conf=args.conf,
		imgsz=args.imgsz,
		padding_ratio=args.padding_ratio,
		output_size=args.output_size,
		classes=[0],
	)

	images_seen = 0
	preview_saved = 0
	detections_seen = 0
	warped_candidates_seen = 0
	contour_warped_saved = 0
	warped_saved = 0

	for source in args.source:
		for image_path in list_image_files(source):
			images_seen += 1
			image_bgr = cv2.imread(str(image_path))
			if image_bgr is None:
				raise RuntimeError(f"failed to read image: {image_path}")

			detections = warper.detect(image_bgr)
			preview_bgr = draw_detection_preview(image_bgr, detections)
			preview_path = preview_dir / f"{image_path.stem}_preview.jpg"
			cv2.imwrite(str(preview_path), preview_bgr)
			preview_saved += 1

			if detections:
				best_detection = max(detections, key=lambda detection: detection["confidence"])
				items = [warper.warp_detection(image_bgr, best_detection, index=0)]
			else:
				items = []
			detections_seen += len(items)
			warped_candidates_seen += len(detections)

			for item in items:
				stem = f"{image_path.stem}_det{item.index:02d}"
				if args.save_contour_warped:
					contour_warped_bgr = warp_screen_from_crop(
						item.crop_bgr,
						min_area_ratio=warper.contour_min_area_ratio,
						target_aspect_ratio=warper.contour_target_aspect_ratio,
						aspect_ratio_tolerance=warper.contour_aspect_ratio_tolerance,
					)
					if contour_warped_bgr is not None:
						contour_warped_path = contour_warped_dir / f"{stem}_contour_warped.jpg"
						cv2.imwrite(str(contour_warped_path), contour_warped_bgr)
						contour_warped_saved += 1
				warped_path = warped_dir / f"{stem}_warped.jpg"
				cv2.imwrite(str(warped_path), item.warped_bgr)
				if args.save_crops:
					crop_path = crop_dir / f"{stem}_crop.jpg"
					cv2.imwrite(str(crop_path), item.crop_bgr)
				warped_saved += 1

	summary = SampleWarpSummary(
		run_timestamp=run_ts,
		weights=str(args.weights),
		sources=[str(source) for source in args.source],
		output_dir=str(output_dir),
		images_seen=images_seen,
		preview_saved=preview_saved,
		detections_seen=detections_seen,
		warped_candidates_seen=warped_candidates_seen,
		contour_warped_saved=contour_warped_saved,
		warped_saved=warped_saved,
		device=args.device,
		confidence_threshold=args.conf,
		image_size=args.imgsz,
	)
	(output_dir / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
	return summary


def main() -> None:
	args = parse_args()
	summary = run_sample_test(args)
	print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()