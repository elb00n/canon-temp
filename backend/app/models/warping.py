from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.models.target_model import resolve_torch_device

YOLO = None


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = BASE_DIR / "outputs" / "yolo_runs" / "screen_train-2" / "weights" / "best.pt"
DEFAULT_OUTPUT_SIZE = 640
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CONTOUR_MIN_AREA_RATIO = 0.70
DEFAULT_CONTOUR_TARGET_ASPECT_RATIO = 1.8
DEFAULT_CONTOUR_ASPECT_RATIO_TOLERANCE = 0.45


@dataclass(slots=True)
class WarpedDetection:
	index: int
	class_id: int
	class_name: str
	confidence: float
	bbox_xyxy: tuple[float, float, float, float]
	crop_bgr: np.ndarray
	warped_bgr: np.ndarray

	def as_dict(self) -> dict[str, object]:
		return {
			"index": self.index,
			"class_id": self.class_id,
			"class_name": self.class_name,
			"confidence": self.confidence,
			"bbox_xyxy": list(self.bbox_xyxy),
		}


def list_image_files(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def order_points(points: np.ndarray) -> np.ndarray:
	ordered = np.zeros((4, 2), dtype=np.float32)
	summed = points.sum(axis=1)
	diff = np.diff(points, axis=1)
	ordered[0] = points[np.argmin(summed)]
	ordered[2] = points[np.argmax(summed)]
	ordered[1] = points[np.argmin(diff)]
	ordered[3] = points[np.argmax(diff)]
	return ordered


def load_bgr_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"failed to read image: {image_path}")
	return image


def clip_bbox(bbox_xyxy: np.ndarray, width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
	x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
	box_width = x2 - x1
	box_height = y2 - y1
	padding_x = box_width * padding_ratio
	padding_y = box_height * padding_ratio
	x1 = max(0.0, x1 - padding_x)
	y1 = max(0.0, y1 - padding_y)
	x2 = min(float(width - 1), x2 + padding_x)
	y2 = min(float(height - 1), y2 + padding_y)
	return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def crop_with_padding(image_bgr: np.ndarray, bbox_xyxy: np.ndarray, padding_ratio: float) -> np.ndarray:
	height, width = image_bgr.shape[:2]
	x1, y1, x2, y2 = clip_bbox(bbox_xyxy, width, height, padding_ratio)
	if x2 <= x1 or y2 <= y1:
		return image_bgr.copy()
	return image_bgr[y1:y2, x1:x2].copy()


def contour_area_ratio(contour: np.ndarray, image_area: float) -> float:
	if image_area <= 0:
		return 0.0
	return float(cv2.contourArea(contour)) / image_area


def contour_aspect_ratio(contour: np.ndarray) -> float:
	x, y, width, height = cv2.boundingRect(contour)
	if width <= 0 or height <= 0:
		return 0.0
	return max(width, height) / max(1, min(width, height))


def contour_border_distance_ratio(contour: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> float:
	if len(image_shape) == 3:
		height, width = image_shape[:2]
	else:
		height, width = image_shape
	x, y, contour_width, contour_height = cv2.boundingRect(contour)
	distance_to_border = min(x, y, width - (x + contour_width), height - (y + contour_height))
	return max(0.0, float(distance_to_border)) / float(max(width, height, 1))


def contour_is_plausible(
	contour: np.ndarray,
	image_shape: tuple[int, int, int] | tuple[int, int],
	*,
	min_area_ratio: float,
	target_aspect_ratio: float,
	aspect_ratio_tolerance: float,
) -> bool:
	image_area = float(image_shape[0] * image_shape[1])
	area_ratio = contour_area_ratio(contour, image_area)
	if area_ratio < min_area_ratio:
		return False
	aspect_ratio = contour_aspect_ratio(contour)
	if aspect_ratio <= 0.0:
		return False
	return abs(aspect_ratio - target_aspect_ratio) <= aspect_ratio_tolerance


def find_best_screen_contour(
	image_bgr: np.ndarray,
	*,
	min_area_ratio: float,
	target_aspect_ratio: float,
	aspect_ratio_tolerance: float,
) -> np.ndarray | None:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(gray, 50, 150)
	kernel = np.ones((3, 3), dtype=np.uint8)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
	edges = cv2.dilate(edges, kernel, iterations=1)

	contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
	if not contours:
		return None

	best_contour: np.ndarray | None = None
	best_score = float("-inf")
	image_area = float(image_bgr.shape[0] * image_bgr.shape[1])

	for contour in contours:
		if not contour_is_plausible(
			contour,
			image_bgr.shape,
			min_area_ratio=min_area_ratio,
			target_aspect_ratio=target_aspect_ratio,
			aspect_ratio_tolerance=aspect_ratio_tolerance,
		):
			continue
		area_ratio = contour_area_ratio(contour, image_area)
		aspect_ratio = contour_aspect_ratio(contour)
		border_distance_ratio = contour_border_distance_ratio(contour, image_bgr.shape)
		aspect_penalty = abs(aspect_ratio - target_aspect_ratio) / max(target_aspect_ratio, 1e-6)
		score = (area_ratio * 2.0) - aspect_penalty - (border_distance_ratio * 0.75)
		if score > best_score:
			best_score = score
			best_contour = contour

	return best_contour


def warp_contour_to_rectangle(image_bgr: np.ndarray, contour: np.ndarray, output_size: int) -> np.ndarray:
	rect = cv2.minAreaRect(contour)
	points = cv2.boxPoints(rect).astype(np.float32)
	source_points = order_points(points)

	width_a = np.linalg.norm(source_points[2] - source_points[3])
	width_b = np.linalg.norm(source_points[1] - source_points[0])
	height_a = np.linalg.norm(source_points[1] - source_points[2])
	height_b = np.linalg.norm(source_points[0] - source_points[3])
	warped_width = max(1, int(round(max(width_a, width_b))))
	warped_height = max(1, int(round(max(height_a, height_b))))

	destination_points = np.array(
		[
			[0.0, 0.0],
			[warped_width - 1.0, 0.0],
			[warped_width - 1.0, warped_height - 1.0],
			[0.0, warped_height - 1.0],
		],
		dtype=np.float32,
	)
	transform = cv2.getPerspectiveTransform(source_points, destination_points)
	return cv2.warpPerspective(
		image_bgr,
		transform,
		(warped_width, warped_height),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REPLICATE,
	)


def warp_screen_from_crop(
	crop_bgr: np.ndarray,
	*,
	min_area_ratio: float,
	target_aspect_ratio: float,
	aspect_ratio_tolerance: float,
) -> np.ndarray | None:
	contour = find_best_screen_contour(
		crop_bgr,
		min_area_ratio=min_area_ratio,
		target_aspect_ratio=target_aspect_ratio,
		aspect_ratio_tolerance=aspect_ratio_tolerance,
	)
	if contour is None:
		return None
	return warp_contour_to_rectangle(crop_bgr, contour, DEFAULT_OUTPUT_SIZE)


def flatten_screen(crop_bgr: np.ndarray, output_size: int = DEFAULT_OUTPUT_SIZE) -> np.ndarray:
	if crop_bgr.size == 0:
		return np.zeros((output_size, output_size, 3), dtype=np.uint8)
	return cv2.resize(crop_bgr, (output_size, output_size))


class YoloScreenWarper:
	def __init__(
		self,
		weights: Path | str = DEFAULT_WEIGHTS,
		*,
		device: str = "cpu",
		conf: float = 0.25,
		imgsz: int = 640,
		padding_ratio: float = 0.02,
		output_size: int = DEFAULT_OUTPUT_SIZE,
		contour_min_area_ratio: float = DEFAULT_CONTOUR_MIN_AREA_RATIO,
		contour_target_aspect_ratio: float = DEFAULT_CONTOUR_TARGET_ASPECT_RATIO,
		contour_aspect_ratio_tolerance: float = DEFAULT_CONTOUR_ASPECT_RATIO_TOLERANCE,
		classes: list[int] | None = None,
	) -> None:
		global YOLO
		if YOLO is None:
			yolo_config_dir = BASE_DIR / "data" / "ultralytics"
			yolo_config_dir.mkdir(parents=True, exist_ok=True)
			os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
			try:
				from ultralytics import YOLO as UltralyticsYOLO
			except ImportError as exc:  # pragma: no cover - optional for mock smoke runs
				raise RuntimeError("ultralytics is required for real YOLO detector inference") from exc
			YOLO = UltralyticsYOLO
		self.weights = Path(weights)
		if not self.weights.exists():
			raise FileNotFoundError(f"weights not found: {self.weights}")
		self.model = YOLO(str(self.weights))
		self.device = resolve_torch_device(device)
		self.conf = conf
		self.imgsz = imgsz
		self.padding_ratio = padding_ratio
		self.output_size = output_size
		self.contour_min_area_ratio = contour_min_area_ratio
		self.contour_target_aspect_ratio = contour_target_aspect_ratio
		self.contour_aspect_ratio_tolerance = contour_aspect_ratio_tolerance
		self.classes = set(classes) if classes is not None else None
		self.class_names = {int(key): value for key, value in getattr(self.model, "names", {}).items()}

	def process_path(self, image_path: Path) -> list[WarpedDetection]:
		return self.process(load_bgr_image(image_path))

	def process_directory(self, root: Path) -> list[WarpedDetection]:
		results: list[WarpedDetection] = []
		for image_path in list_image_files(root):
			results.extend(self.process_path(image_path))
		return results

	def detect(self, image_bgr: np.ndarray) -> list[dict[str, object]]:
		results = self.model.predict(
			source=image_bgr,
			conf=self.conf,
			imgsz=self.imgsz,
			device=self.device,
			verbose=False,
		)
		if not results:
			return []

		result = results[0]
		boxes = getattr(result, "boxes", None)
		if boxes is None or len(boxes) == 0:
			return []

		detections: list[dict[str, object]] = []
		for box in boxes:
			class_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
			if self.classes is not None and class_id not in self.classes:
				continue
			confidence = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
			bbox_xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
			detections.append(
				{
					"class_id": class_id,
					"class_name": self.class_names.get(class_id, str(class_id)),
					"confidence": confidence,
					"bbox_xyxy": bbox_xyxy,
				}
			)
		return detections

	def warp_detection(self, image_bgr: np.ndarray, detection: dict[str, object], index: int = 0) -> WarpedDetection:
		bbox_xyxy = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
		crop_bgr = crop_with_padding(image_bgr, bbox_xyxy, self.padding_ratio)
		warped_crop = warp_screen_from_crop(
			crop_bgr,
			min_area_ratio=self.contour_min_area_ratio,
			target_aspect_ratio=self.contour_target_aspect_ratio,
			aspect_ratio_tolerance=self.contour_aspect_ratio_tolerance,
		)
		warped_bgr = flatten_screen(warped_crop if warped_crop is not None else crop_bgr, self.output_size)
		return WarpedDetection(
			index=index,
			class_id=int(detection["class_id"]),
			class_name=str(detection["class_name"]),
			confidence=float(detection["confidence"]),
			bbox_xyxy=(float(bbox_xyxy[0]), float(bbox_xyxy[1]), float(bbox_xyxy[2]), float(bbox_xyxy[3])),
			crop_bgr=crop_bgr,
			warped_bgr=warped_bgr,
		)

	def process(self, image_bgr: np.ndarray) -> list[WarpedDetection]:
		detections = self.detect(image_bgr)
		if not detections:
			return []
		best_detection = max(detections, key=lambda detection: detection["confidence"])
		return [self.warp_detection(image_bgr, best_detection, index=0)]

	def process_first(self, image_bgr: np.ndarray) -> WarpedDetection | None:
		items = self.process(image_bgr)
		return items[0] if items else None

