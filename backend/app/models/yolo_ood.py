from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.models.target_model import resolve_torch_device

try:
	from ultralytics import YOLO
except ImportError:  # pragma: no cover
	YOLO = None


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = BASE_DIR / "images" / "yolo_dataset"
DEFAULT_AUG_DATASET_ROOT = BASE_DIR / "outputs" / "yolo_aug_dataset"
DEFAULT_DATA_YAML = DEFAULT_DATASET_ROOT / "data.yaml"
DEFAULT_RUN_ROOT = BASE_DIR / "outputs" / "yolo_runs"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class AugmentConfig:
	copies_per_image: int = 4
	rotate_min: float = 3.0
	rotate_max: float = 15.0
	shear_max: float = 8.0
	perspective_max: float = 0.06
	blur_prob: float = 0.75
	glare_prob: float = 0.6
	brightness_prob: float = 0.8
	brightness_alpha: tuple[float, float] = (0.78, 1.22)
	brightness_beta: tuple[int, int] = (-18, 18)
	min_box_area_ratio: float = 0.001


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="YOLO augmentation and training runner")
	subparsers = parser.add_subparsers(dest="command", required=True)

	augment_parser = subparsers.add_parser("augment", help="augment train split only")
	augment_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	augment_parser.add_argument("--output-root", type=Path, default=DEFAULT_AUG_DATASET_ROOT)
	augment_parser.add_argument("--copies-per-image", type=int, default=4)
	augment_parser.add_argument("--seed", type=int, default=42)

	train_parser = subparsers.add_parser("train", help="train YOLO")
	train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
	train_parser.add_argument("--weights", type=str, default="yolov11n.pt")
	train_parser.add_argument("--epochs", type=int, default=80)
	train_parser.add_argument("--imgsz", type=int, default=640)
	train_parser.add_argument("--batch", type=int, default=8)
	train_parser.add_argument("--device", type=str, default="cpu")
	train_parser.add_argument("--project", type=Path, default=DEFAULT_RUN_ROOT)
	train_parser.add_argument("--name", type=str, default="screen_train")

	eval_parser = subparsers.add_parser("eval", help="evaluate YOLO")
	eval_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
	eval_parser.add_argument("--weights", type=Path, required=True)
	eval_parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
	eval_parser.add_argument("--imgsz", type=int, default=640)
	eval_parser.add_argument("--device", type=str, default="cpu")
	eval_parser.add_argument("--project", type=Path, default=DEFAULT_RUN_ROOT)
	eval_parser.add_argument("--name", type=str, default="screen_eval")

	all_parser = subparsers.add_parser("all", help="augment -> train -> eval")
	all_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
	all_parser.add_argument("--output-root", type=Path, default=DEFAULT_AUG_DATASET_ROOT)
	all_parser.add_argument("--copies-per-image", type=int, default=4)
	all_parser.add_argument("--seed", type=int, default=42)
	all_parser.add_argument("--weights", type=str, default="yolov11n.pt")
	all_parser.add_argument("--epochs", type=int, default=80)
	all_parser.add_argument("--imgsz", type=int, default=640)
	all_parser.add_argument("--batch", type=int, default=8)
	all_parser.add_argument("--device", type=str, default="cpu")
	all_parser.add_argument("--project", type=Path, default=DEFAULT_RUN_ROOT)
	all_parser.add_argument("--name", type=str, default="screen_train")
	all_parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"])

	return parser.parse_args()


def require_ultralytics() -> None:
	if YOLO is None:
		raise RuntimeError("ultralytics가 설치되어 있지 않습니다. `uv add ultralytics` 후 다시 실행하세요.")


def list_images(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
	if not label_path.exists():
		return []
	rows: list[tuple[int, float, float, float, float]] = []
	for line in label_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		parts = line.split()
		if len(parts) < 5:
			continue
		rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
	return rows


def write_yolo_labels(label_path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
	label_path.parent.mkdir(parents=True, exist_ok=True)
	content = "\n".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in rows)
	label_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def image_to_label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
	relative = image_path.relative_to(images_root)
	return labels_root / relative.with_suffix(".txt")


def yolo_to_xyxy(box: tuple[int, float, float, float, float], width: int, height: int) -> np.ndarray:
	_, x_center, y_center, box_width, box_height = box
	x1 = (x_center - box_width / 2.0) * width
	y1 = (y_center - box_height / 2.0) * height
	x2 = (x_center + box_width / 2.0) * width
	y2 = (y_center + box_height / 2.0) * height
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_yolo(xyxy: np.ndarray, width: int, height: int, min_area_ratio: float) -> tuple[float, float, float, float] | None:
	x1, y1, x2, y2 = xyxy
	x1 = max(0.0, min(float(width - 1), x1))
	y1 = max(0.0, min(float(height - 1), y1))
	x2 = max(0.0, min(float(width - 1), x2))
	y2 = max(0.0, min(float(height - 1), y2))
	if x2 <= x1 or y2 <= y1:
		return None
	box_width = (x2 - x1) / width
	box_height = (y2 - y1) / height
	if box_width * box_height < min_area_ratio:
		return None
	x_center = ((x1 + x2) / 2.0) / width
	y_center = ((y1 + y2) / 2.0) / height
	return x_center, y_center, box_width, box_height


def transform_boxes_affine(boxes: list[tuple[int, float, float, float, float]], matrix: np.ndarray, width: int, height: int, min_area_ratio: float) -> list[tuple[int, float, float, float, float]]:
	transformed: list[tuple[int, float, float, float, float]] = []
	for box in boxes:
		xyxy = yolo_to_xyxy(box, width, height)
		corners = np.array([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]], dtype=np.float32).reshape(-1, 1, 2)
		warped = cv2.transform(corners, matrix).reshape(-1, 2)
		new_box = np.array([warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()], dtype=np.float32)
		yolo_box = xyxy_to_yolo(new_box, width, height, min_area_ratio)
		if yolo_box is not None:
			transformed.append((box[0], *yolo_box))
	return transformed


def transform_boxes_perspective(boxes: list[tuple[int, float, float, float, float]], matrix: np.ndarray, width: int, height: int, min_area_ratio: float) -> list[tuple[int, float, float, float, float]]:
	transformed: list[tuple[int, float, float, float, float]] = []
	for box in boxes:
		xyxy = yolo_to_xyxy(box, width, height)
		corners = np.array([[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]], dtype=np.float32).reshape(-1, 1, 2)
		warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
		new_box = np.array([warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max()], dtype=np.float32)
		yolo_box = xyxy_to_yolo(new_box, width, height, min_area_ratio)
		if yolo_box is not None:
			transformed.append((box[0], *yolo_box))
	return transformed


def apply_rotate(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], cfg: AugmentConfig, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
	height, width = image.shape[:2]
	angle = rng.choice([-1.0, 1.0]) * rng.uniform(cfg.rotate_min, cfg.rotate_max)
	matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
	rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return rotated, transform_boxes_affine(boxes, matrix, width, height, cfg.min_box_area_ratio)


def apply_shear(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], cfg: AugmentConfig, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
	height, width = image.shape[:2]
	shear_x = rng.uniform(-cfg.shear_max, cfg.shear_max)
	shear_y = rng.uniform(-cfg.shear_max / 2.0, cfg.shear_max / 2.0)
	matrix = np.array([[1.0, math.tan(math.radians(shear_x)), 0.0], [math.tan(math.radians(shear_y)), 1.0, 0.0]], dtype=np.float32)
	sheared = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return sheared, transform_boxes_affine(boxes, matrix, width, height, cfg.min_box_area_ratio)


def apply_perspective(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], cfg: AugmentConfig, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
	height, width = image.shape[:2]
	max_offset = int(min(width, height) * cfg.perspective_max)
	src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
	dst = np.float32([
		[rng.randint(-max_offset, max_offset), rng.randint(-max_offset, max_offset)],
		[width - 1 + rng.randint(-max_offset, max_offset), rng.randint(-max_offset, max_offset)],
		[width - 1 + rng.randint(-max_offset, max_offset), height - 1 + rng.randint(-max_offset, max_offset)],
		[rng.randint(-max_offset, max_offset), height - 1 + rng.randint(-max_offset, max_offset)],
	])
	matrix = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return warped, transform_boxes_perspective(boxes, matrix, width, height, cfg.min_box_area_ratio)


def apply_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
	choice = rng.random()
	if choice < 0.45:
		kernel = rng.choice([3, 5, 7])
		return cv2.GaussianBlur(image, (kernel, kernel), 0)
	if choice < 0.8:
		kernel = rng.choice([3, 5, 7])
		motion = np.zeros((kernel, kernel), dtype=np.float32)
		motion[kernel // 2, :] = 1.0 / kernel
		angle = rng.uniform(0, 180)
		rot = cv2.getRotationMatrix2D((kernel / 2.0, kernel / 2.0), angle, 1.0)
		motion = cv2.warpAffine(motion, rot, (kernel, kernel))
		motion /= max(1e-6, motion.sum())
		return cv2.filter2D(image, -1, motion)
	return cv2.medianBlur(image, 3)


def apply_glare_and_brightness(image: np.ndarray, cfg: AugmentConfig, rng: random.Random) -> np.ndarray:
	out = image.copy()
	if rng.random() < cfg.brightness_prob:
		alpha = rng.uniform(*cfg.brightness_alpha)
		beta = rng.randint(*cfg.brightness_beta)
		out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

	if rng.random() < cfg.glare_prob:
		height, width = out.shape[:2]
		overlay = np.zeros((height, width), dtype=np.float32)
		for _ in range(rng.randint(1, 3)):
			center_x = rng.randint(0, width - 1)
			center_y = rng.randint(0, height - 1)
			radius = rng.randint(max(20, min(width, height) // 8), max(40, min(width, height) // 3))
			intensity = rng.uniform(0.35, 0.9)
			mask = np.zeros((height, width), dtype=np.float32)
			cv2.circle(mask, (center_x, center_y), radius, intensity, -1)
			mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 2.0, sigmaY=radius / 2.0)
			overlay = np.maximum(overlay, mask)
		out = np.clip(out.astype(np.float32) + overlay[:, :, None] * 180.0, 0, 255).astype(np.uint8)

	return out


def augment_sample(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], cfg: AugmentConfig, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
	out_image = image
	out_boxes = boxes
	out_image, out_boxes = apply_rotate(out_image, out_boxes, cfg, rng)
	out_image, out_boxes = apply_shear(out_image, out_boxes, cfg, rng)
	out_image, out_boxes = apply_perspective(out_image, out_boxes, cfg, rng)
	out_image = apply_blur(out_image, rng)
	out_image = apply_glare_and_brightness(out_image, cfg, rng)
	return out_image, out_boxes


def augment_dataset(dataset_root: Path, output_root: Path, cfg: AugmentConfig, seed: int) -> Path:
	rng = random.Random(seed)
	if output_root.exists():
		shutil.rmtree(output_root)

	for split in ("val", "test"):
		images_src = dataset_root / "images" / split
		labels_src = dataset_root / "labels" / split
		images_dst = output_root / "images" / split
		labels_dst = output_root / "labels" / split
		images_dst.mkdir(parents=True, exist_ok=True)
		labels_dst.mkdir(parents=True, exist_ok=True)
		for image_path in list_images(images_src):
			label_path = image_to_label_path(image_path, images_src, labels_src)
			shutil.copy2(image_path, images_dst / image_path.name)
			if label_path.exists():
				shutil.copy2(label_path, labels_dst / label_path.name)

	train_images_src = dataset_root / "images" / "train"
	train_labels_src = dataset_root / "labels" / "train"
	train_images_dst = output_root / "images" / "train"
	train_labels_dst = output_root / "labels" / "train"
	train_images_dst.mkdir(parents=True, exist_ok=True)
	train_labels_dst.mkdir(parents=True, exist_ok=True)

	for image_path in list_images(train_images_src):
		label_path = image_to_label_path(image_path, train_images_src, train_labels_src)
		if not label_path.exists():
			continue
		image = cv2.imread(str(image_path))
		if image is None:
			continue
		boxes = read_yolo_labels(label_path)
		if not boxes:
			continue

		shutil.copy2(image_path, train_images_dst / image_path.name)
		shutil.copy2(label_path, train_labels_dst / label_path.name)

		for index in range(cfg.copies_per_image):
			aug_image, aug_boxes = augment_sample(image, boxes, cfg, rng)
			if not aug_boxes:
				continue
			stem = f"{image_path.stem}_aug{index:02d}"
			out_image_path = train_images_dst / f"{stem}{image_path.suffix}"
			out_label_path = train_labels_dst / f"{stem}.txt"
			cv2.imwrite(str(out_image_path), aug_image)
			write_yolo_labels(out_label_path, aug_boxes)

	yaml_src = dataset_root / "data.yaml"
	yaml_dst = output_root / "data.yaml"
	if yaml_src.exists():
		shutil.copy2(yaml_src, yaml_dst)

	return output_root


def train_yolo(data_yaml: Path, weights: str, epochs: int, imgsz: int, batch: int, device: str, project: Path, name: str):
	require_ultralytics()
	model = YOLO(weights)
	resolved_device = resolve_torch_device(device)
	return model.train(
		data=str(data_yaml),
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		device=resolved_device,
		project=str(project),
		name=name,
		patience=20,
		mosaic=0.0,
		mixup=0.0,
		copy_paste=0.0,
		fliplr=0.0,
		flipud=0.0,
		degrees=0.0,
		translate=0.0,
		scale=0.0,
		shear=0.0,
		perspective=0.0,
		hsv_h=0.015,
		hsv_s=0.5,
		hsv_v=0.35,
	)


def evaluate_yolo(weights: Path, data_yaml: Path, split: str, imgsz: int, device: str, project: Path, name: str):
	require_ultralytics()
	model = YOLO(str(weights))
	resolved_device = resolve_torch_device(device)
	return model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=resolved_device, project=str(project), name=name)


def main() -> None:
	args = parse_args()

	if args.command == "augment":
		cfg = AugmentConfig(copies_per_image=args.copies_per_image)
		out_root = augment_dataset(args.dataset_root, args.output_root, cfg, args.seed)
		print(json.dumps({"output_root": str(out_root)}, indent=2, ensure_ascii=False))
		return

	if args.command == "train":
		results = train_yolo(args.data, args.weights, args.epochs, args.imgsz, args.batch, args.device, args.project, args.name)
		print(json.dumps(getattr(results, "results_dict", {}), indent=2, ensure_ascii=False))
		return

	if args.command == "eval":
		results = evaluate_yolo(args.weights, args.data, args.split, args.imgsz, args.device, args.project, args.name)
		print(json.dumps(getattr(results, "results_dict", {}), indent=2, ensure_ascii=False))
		return

	if args.command == "all":
		cfg = AugmentConfig(copies_per_image=args.copies_per_image)
		aug_root = augment_dataset(args.dataset_root, args.output_root, cfg, args.seed)
		train_results = train_yolo(aug_root / "data.yaml", args.weights, args.epochs, args.imgsz, args.batch, args.device, args.project, args.name)
		print(json.dumps(getattr(train_results, "results_dict", {}), indent=2, ensure_ascii=False))
		best_weights = args.project / args.name / "weights" / "best.pt"
		if best_weights.exists():
			eval_results = evaluate_yolo(best_weights, aug_root / "data.yaml", args.eval_split, args.imgsz, args.device, args.project, f"{args.name}_eval")
			print(json.dumps(getattr(eval_results, "results_dict", {}), indent=2, ensure_ascii=False))
		return


if __name__ == "__main__":
	main()

