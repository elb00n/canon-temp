"""Target test service for YOLO detection + warping + target classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from app.core.paths import ASSET_WEIGHTS_DIR, OUTPUTS_DIR, TARGET_TEST_RUNS_DIR, yolo_weight_file
from app.models.target_model import DEFAULT_THRESHOLD, get_target_threshold
from app.models.warping import YoloScreenWarper, list_image_files, warp_screen_from_crop
from app.service.target_service import TargetService


@dataclass(slots=True)
class TargetTestRunConfig:
	target_name: str
	target_root: Path = ASSET_WEIGHTS_DIR
	yolo_weights: Path = yolo_weight_file()
	source: list[Path] = field(default_factory=lambda: [OUTPUTS_DIR / "warped"])
	output_dir: Path = TARGET_TEST_RUNS_DIR
	device: str = "cpu"
	threshold: float | None = None
	conf: float = 0.25
	imgsz: int = 640
	padding_ratio: float = 0.02
	save_crops: bool = False
	save_contour_warped: bool = False


@dataclass(slots=True)
class TargetTestSummary:
	run_timestamp: str
	target_name: str
	target_model_weights: str
	yolo_weights: str
	sources: list[str]
	output_dir: str
	images_seen: int
	preview_saved: int
	detections_seen: int
	yes_saved: int
	no_saved: int
	no_detection_saved: int
	contour_warped_saved: int
	device: str
	threshold: float
	confidence_threshold: float
	image_size: int


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def draw_detection_preview(image_bgr, detections):
	preview = image_bgr.copy()
	for detection in detections:
		bbox_xyxy = detection["bbox_xyxy"]
		x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox_xyxy]
		label = f'{detection["class_name"]} {float(detection["confidence"]):.2f}'
		cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
		label_y = max(18, y_min - 8)
		cv2.putText(preview, label, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
	return preview


class TargetTestService:
	"""Service layer for sample-folder target testing."""

	def __init__(self, config: TargetTestRunConfig) -> None:
		self.config = config
		self.target_service = TargetService(target_root=config.target_root, device=config.device, prefer_openvino=True)

	def resolve_target_weights(self) -> Path:
		return self.target_service.get_handle(self.config.target_name).weights

	def run(self) -> dict[str, object]:
		for source in self.config.source:
			if not source.exists():
				raise FileNotFoundError(f"source not found: {source}")
		if not self.config.yolo_weights.exists():
			raise FileNotFoundError(f"YOLO weights not found: {self.config.yolo_weights}")

		target_handle = self.target_service.get_handle(self.config.target_name)
		threshold = self.config.threshold if self.config.threshold is not None else get_target_threshold(self.config.target_name, DEFAULT_THRESHOLD)
		run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir = self.config.output_dir / f"{self.config.target_name}_{run_ts}"
		preview_dir = run_dir / "preview"
		yes_dir = run_dir / "yes"
		no_dir = run_dir / "no"
		no_detection_dir = run_dir / "no_detection"
		crop_dir = run_dir / "crops"
		contour_warped_dir = run_dir / "contour_warped"
		for path in (preview_dir, yes_dir, no_dir, no_detection_dir):
			ensure_dir(path)
		if self.config.save_crops:
			ensure_dir(crop_dir)
		if self.config.save_contour_warped:
			ensure_dir(contour_warped_dir)

		warper = YoloScreenWarper(
			weights=self.config.yolo_weights,
			device=self.config.device,
			conf=self.config.conf,
			imgsz=self.config.imgsz,
			padding_ratio=self.config.padding_ratio,
			output_size=640,
			classes=[0],
		)

		images_seen = 0
		preview_saved = 0
		detections_seen = 0
		yes_saved = 0
		no_saved = 0
		no_detection_saved = 0
		contour_warped_saved = 0

		for source in self.config.source:
			for image_path in list_image_files(source):
				images_seen += 1
				image_bgr = cv2.imread(str(image_path))
				if image_bgr is None:
					raise RuntimeError(f"failed to read image: {image_path}")

				detections = warper.detect(image_bgr)
				preview_bgr = draw_detection_preview(image_bgr, detections)
				cv2.imwrite(str(preview_dir / f"{image_path.stem}_preview.jpg"), preview_bgr)
				preview_saved += 1

				if not detections:
					cv2.imwrite(str(no_detection_dir / f"{image_path.stem}_no_detection.jpg"), image_bgr)
					no_detection_saved += 1
					continue

				best_detection = max(detections, key=lambda detection: detection["confidence"])
				warped_detection = warper.warp_detection(image_bgr, best_detection, index=0)
				detections_seen += len(detections)

				if self.config.save_crops:
					cv2.imwrite(str(crop_dir / f"{image_path.stem}_crop.jpg"), warped_detection.crop_bgr)

				contour_warped_bgr = warp_screen_from_crop(
					warped_detection.crop_bgr,
					min_area_ratio=warper.contour_min_area_ratio,
					target_aspect_ratio=warper.contour_target_aspect_ratio,
					aspect_ratio_tolerance=warper.contour_aspect_ratio_tolerance,
				)
				if contour_warped_bgr is not None and self.config.save_contour_warped:
					cv2.imwrite(str(contour_warped_dir / f"{image_path.stem}_contour_warped.jpg"), contour_warped_bgr)
					contour_warped_saved += 1

				prediction = target_handle.predict_bgr(warped_detection.warped_bgr, device=self.config.device, threshold=threshold)
				target_dir = yes_dir if prediction.label == "yes" else no_dir
				output_name = f"{image_path.stem}_det{warped_detection.index:02d}_{prediction.label}_{prediction.score:.3f}.jpg"
				cv2.imwrite(str(target_dir / output_name), warped_detection.warped_bgr)
				if prediction.label == "yes":
					yes_saved += 1
				else:
					no_saved += 1

		summary = TargetTestSummary(
			run_timestamp=run_ts,
			target_name=self.config.target_name,
			target_model_weights=str(target_handle.weights),
			yolo_weights=str(self.config.yolo_weights),
			sources=[str(source) for source in self.config.source],
			output_dir=str(run_dir),
			images_seen=images_seen,
			preview_saved=preview_saved,
			detections_seen=detections_seen,
			yes_saved=yes_saved,
			no_saved=no_saved,
			no_detection_saved=no_detection_saved,
			contour_warped_saved=contour_warped_saved,
			device=self.config.device,
			threshold=threshold,
			confidence_threshold=self.config.conf,
			image_size=self.config.imgsz,
		)
		(run_dir / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
		return {**asdict(summary)}


__all__ = [
	"TargetTestRunConfig",
	"TargetTestService",
	"TargetTestSummary",
	"draw_detection_preview",
	"ensure_dir",
]