from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.models.warping import YoloScreenWarper, list_image_files, warp_screen_from_crop


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = BASE_DIR / "images" / "target_image"
DEFAULT_OUTPUT_DIR = BASE_DIR / "images" / "agumentation_images"
DEFAULT_OUTPUT_SIZE = 640
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class AugmentConfig:
	copies_per_image: int = 10
	output_size: int = DEFAULT_OUTPUT_SIZE
	rotate_min: float = 3.0
	rotate_max: float = 15.0
	shear_max: float = 6.0
	perspective_max: float = 0.05
	blur_prob: float = 0.75
	glare_prob: float = 0.60
	brightness_prob: float = 0.85
	brightness_alpha: tuple[float, float] = (0.75, 1.25)
	brightness_beta: tuple[int, int] = (-20, 20)
	noise_prob: float = 0.55
	occlusion_prob: float = 0.25
	min_occlusion_ratio: float = 0.03
	max_occlusion_ratio: float = 0.12
	min_box_area_ratio: float = 0.001


@dataclass(slots=True)
class AugmentSummary:
	run_timestamp: str
	source_dir: str
	output_root: str
	run_dir: str
	source_classes: list[str]
	output_size: int
	images_seen: int
	previews_saved: int
	detections_seen: int
	contour_warped_saved: int
	warped_saved: int
	augmented_saved: int
	fallback_saved: int
	copies_per_image: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Warp, augment, and export target images into per-class folders")
	parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="원본 target 이미지 폴더")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="증강 결과 폴더")
	parser.add_argument("--copies-per-image", type=int, default=10, help="원본당 증강 이미지 수")
	parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="최종 리사이즈 크기")
	parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
	parser.add_argument("--keep-existing", action="store_true", help="기존 output-dir 내용을 삭제하지 않음")
	parser.add_argument("--weights", type=Path, default=BASE_DIR / "outputs" / "yolo_runs" / "screen_train-2" / "weights" / "best.pt")
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--padding-ratio", type=float, default=0.02)
	parser.add_argument("--save-crops", action="store_true", help="Save raw crops for target classes")
	parser.add_argument("--save-contour-warped", action="store_true", help="Save contour-warped images for target classes")
	return parser.parse_args()


def list_image_files(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def list_source_class_dirs(root: Path) -> list[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.iterdir() if path.is_dir())


def load_bgr_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"failed to read image: {image_path}")
	return image


def pad_to_square(image_bgr: np.ndarray, fill_value: int = 0) -> np.ndarray:
	if image_bgr.size == 0:
		return np.zeros((1, 1, 3), dtype=np.uint8)

	height, width = image_bgr.shape[:2]
	side = max(height, width)
	canvas = np.full((side, side, 3), fill_value, dtype=image_bgr.dtype)
	top = (side - height) // 2
	left = (side - width) // 2
	canvas[top : top + height, left : left + width] = image_bgr
	return canvas


def resize_for_training(image_bgr: np.ndarray, output_size: int) -> np.ndarray:
	letterboxed = pad_to_square(image_bgr)
	return cv2.resize(letterboxed, (output_size, output_size), interpolation=cv2.INTER_AREA)


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


def write_image(output_path: Path, image_bgr: np.ndarray) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(output_path), image_bgr):
		raise RuntimeError(f"failed to write image: {output_path}")


def apply_rotation(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	height, width = image.shape[:2]
	angle = rng.choice([-1.0, 1.0]) * rng.uniform(cfg.rotate_min, cfg.rotate_max)
	matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
	return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_shear(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	height, width = image.shape[:2]
	shear_x = math.tan(math.radians(rng.uniform(-cfg.shear_max, cfg.shear_max)))
	shear_y = math.tan(math.radians(rng.uniform(-cfg.shear_max / 2.0, cfg.shear_max / 2.0)))
	matrix = np.array([[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0]], dtype=np.float32)
	return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_perspective(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	height, width = image.shape[:2]
	max_offset = int(min(width, height) * cfg.perspective_max)
	if max_offset <= 0:
		return image
	src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
	dst = np.float32([
		[rng.randint(-max_offset, max_offset), rng.randint(-max_offset, max_offset)],
		[width - 1 + rng.randint(-max_offset, max_offset), rng.randint(-max_offset, max_offset)],
		[width - 1 + rng.randint(-max_offset, max_offset), height - 1 + rng.randint(-max_offset, max_offset)],
		[rng.randint(-max_offset, max_offset), height - 1 + rng.randint(-max_offset, max_offset)],
	])
	matrix = cv2.getPerspectiveTransform(src, dst)
	return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


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


def apply_brightness_and_contrast(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	if rng.random() > cfg.brightness_prob:
		return image
	alpha = rng.uniform(*cfg.brightness_alpha)
	beta = rng.randint(*cfg.brightness_beta)
	return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_noise(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	if rng.random() > cfg.noise_prob:
		return image
	noise = rng.normalvariate(0.0, 1.0)
	std = rng.uniform(4.0, 18.0)
	gaussian = np.random.normal(noise, std, image.shape).astype(np.float32)
	noisy = np.clip(image.astype(np.float32) + gaussian, 0, 255).astype(np.uint8)
	return noisy


def apply_glare(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	if rng.random() > cfg.glare_prob:
		return image
	out = image.copy().astype(np.float32)
	height, width = out.shape[:2]
	overlay = np.zeros((height, width), dtype=np.float32)
	for _ in range(rng.randint(1, 2)):
		center_x = rng.randint(0, width - 1)
		center_y = rng.randint(0, height - 1)
		radius = rng.randint(max(20, min(width, height) // 10), max(40, min(width, height) // 3))
		intensity = rng.uniform(0.25, 0.85)
		mask = np.zeros((height, width), dtype=np.float32)
		cv2.circle(mask, (center_x, center_y), radius, intensity, -1)
		mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 2.0, sigmaY=radius / 2.0)
		overlay = np.maximum(overlay, mask)
	out = np.clip(out + overlay[:, :, None] * 190.0, 0, 255).astype(np.uint8)
	return out


def apply_occlusion(image: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	if rng.random() > cfg.occlusion_prob:
		return image
	out = image.copy()
	height, width = out.shape[:2]
	occ_w = max(1, int(width * rng.uniform(cfg.min_occlusion_ratio, cfg.max_occlusion_ratio)))
	occ_h = max(1, int(height * rng.uniform(cfg.min_occlusion_ratio, cfg.max_occlusion_ratio)))
	x1 = rng.randint(0, max(0, width - occ_w))
	y1 = rng.randint(0, max(0, height - occ_h))
	color = tuple(int(rng.uniform(0, 35)) for _ in range(3))
	cv2.rectangle(out, (x1, y1), (x1 + occ_w, y1 + occ_h), color, -1)
	return out


def augment_once(image_bgr: np.ndarray, rng: random.Random, cfg: AugmentConfig) -> np.ndarray:
	out = image_bgr.copy()
	steps = [
		lambda img: apply_rotation(img, rng, cfg),
		lambda img: apply_shear(img, rng, cfg),
		lambda img: apply_perspective(img, rng, cfg),
	]
	rng.shuffle(steps)
	for step in steps:
		if rng.random() < 0.8:
			out = step(out)
	out = apply_blur(out, rng)
	out = apply_brightness_and_contrast(out, rng, cfg)
	out = apply_noise(out, rng, cfg)
	out = apply_glare(out, rng, cfg)
	out = apply_occlusion(out, rng, cfg)
	return out


def target_name_from_source(image_path: Path) -> str:
	stem = image_path.stem
	if stem.startswith("target_"):
		return stem
	return f"target_{stem}"


def clear_directory(directory: Path) -> None:
	if not directory.exists():
		return
	for path in list(directory.iterdir()):
		if path.is_dir():
			shutil.rmtree(path)
		else:
			path.unlink()


def prepare_class_dirs(run_dir: Path, class_name: str) -> dict[str, Path]:
	class_root = run_dir / class_name
	paths = {
		"root": class_root,
		"preview": class_root / "preview",
		"warped": class_root / "warped",
		"augmented": class_root / "augmented",
		"crops": class_root / "crops",
		"contour_warped": class_root / "contour_warped",
	}
	for path in paths.values():
		path.mkdir(parents=True, exist_ok=True)
	return paths


def process_target_image(
	image_path: Path,
	*,
	class_name: str,
	paths: dict[str, Path],
	warper: YoloScreenWarper,
	cfg: AugmentConfig,
	rng: random.Random,
	save_crops: bool,
	save_contour_warped: bool,
) -> tuple[int, int, int, int, int]:
	image_bgr = load_bgr_image(image_path)
	preview_bgr = image_bgr.copy()
	detection_count = 0
	contour_saved = 0
	warped_saved = 0
	augmented_saved = 0
	fallback_saved = 0

	if class_name == "others":
		canonical_bgr = resize_for_training(image_bgr, cfg.output_size)
		fallback_saved = 1
	else:
		detections = warper.detect(image_bgr)
		detection_count = len(detections)
		preview_bgr = draw_detection_preview(image_bgr, detections)
		if detections:
			best_detection = max(detections, key=lambda detection: detection["confidence"])
			warped_detection = warper.warp_detection(image_bgr, best_detection, index=0)
			canonical_bgr = warped_detection.warped_bgr
			if save_crops:
				write_image(paths["crops"] / f"{image_path.stem}_crop.jpg", warped_detection.crop_bgr)
			if save_contour_warped:
				contour_warped_bgr = warp_screen_from_crop(
					warped_detection.crop_bgr,
					min_area_ratio=warper.contour_min_area_ratio,
					target_aspect_ratio=warper.contour_target_aspect_ratio,
					aspect_ratio_tolerance=warper.contour_aspect_ratio_tolerance,
				)
				if contour_warped_bgr is not None:
					write_image(paths["contour_warped"] / f"{image_path.stem}_contour_warped.jpg", contour_warped_bgr)
					contour_saved = 1
			warped_saved = 1
		else:
			canonical_bgr = resize_for_training(image_bgr, cfg.output_size)
			fallback_saved = 1

	write_image(paths["preview"] / f"{image_path.stem}_preview.jpg", preview_bgr)
	write_image(paths["warped"] / f"{image_path.stem}_warped.jpg", canonical_bgr)

	for index in range(cfg.copies_per_image):
		aug_image = augment_once(canonical_bgr, rng, cfg)
		aug_image = resize_for_training(aug_image, cfg.output_size)
		write_image(paths["augmented"] / f"{image_path.stem}_aug{index:02d}.jpg", aug_image)
		augmented_saved += 1

	return detection_count, contour_saved, warped_saved, augmented_saved, fallback_saved


def run_augmentation(source_dir: Path, output_root: Path, cfg: AugmentConfig, seed: int, keep_existing: bool, warper: YoloScreenWarper, save_crops: bool, save_contour_warped: bool) -> AugmentSummary:
	if not source_dir.exists():
		raise FileNotFoundError(f"source target folder not found: {source_dir}")

	output_root.mkdir(parents=True, exist_ok=True)
	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = output_root / f"augmentation_{run_ts}"
	if run_dir.exists() and not keep_existing:
		clear_directory(run_dir)
	run_dir.mkdir(parents=True, exist_ok=True)

	rng = random.Random(seed)
	class_dirs = list_source_class_dirs(source_dir)
	source_classes = [class_dir.name for class_dir in class_dirs]
	images_seen = 0
	previews_saved = 0
	detections_seen = 0
	contour_warped_saved = 0
	warped_saved = 0
	augmented_saved = 0
	fallback_saved = 0

	for class_dir in class_dirs:
		class_name = class_dir.name
		paths = prepare_class_dirs(run_dir, class_name)
		for image_path in list_image_files(class_dir):
			images_seen += 1
			detection_count, contour_saved, warped_count, augmented_count, fallback_count = process_target_image(
				image_path,
				class_name=class_name,
				paths=paths,
				warper=warper,
				cfg=cfg,
				rng=rng,
				save_crops=save_crops,
				save_contour_warped=save_contour_warped,
			)
			detections_seen += detection_count
			contour_warped_saved += contour_saved
			warped_saved += warped_count
			augmented_saved += augmented_count
			fallback_saved += fallback_count

	summary = AugmentSummary(
		run_timestamp=run_ts,
		source_dir=str(source_dir),
		output_root=str(output_root),
		run_dir=str(run_dir),
		source_classes=source_classes,
		output_size=cfg.output_size,
		images_seen=images_seen,
		previews_saved=images_seen,
		detections_seen=detections_seen,
		contour_warped_saved=contour_warped_saved,
		warped_saved=warped_saved,
		augmented_saved=augmented_saved,
		fallback_saved=fallback_saved,
		copies_per_image=cfg.copies_per_image,
	)
	(run_dir / "augment_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
	return summary


def main() -> None:
	args = parse_args()
	if not args.weights.exists():
		raise FileNotFoundError(f"weights not found: {args.weights}")
	warper = YoloScreenWarper(
		weights=args.weights,
		device=args.device,
		conf=args.conf,
		imgsz=args.imgsz,
		padding_ratio=args.padding_ratio,
		output_size=args.output_size,
		classes=[0],
	)
	cfg = AugmentConfig(copies_per_image=args.copies_per_image, output_size=args.output_size)
	summary = run_augmentation(
		args.source_dir,
		args.output_dir,
		cfg,
		args.seed,
		args.keep_existing,
		warper,
		args.save_crops,
		args.save_contour_warped,
	)
	print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
