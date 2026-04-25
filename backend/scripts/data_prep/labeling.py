from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOTS = [BASE_DIR / "sample_images", BASE_DIR / "sample_images_from_videos"]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "yolo_dataset"
DEFAULT_RUN_ROOT = BASE_DIR.parent / "outputs" / "labeling_runs"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "screen"
MIN_SCREEN_AREA_RATIO = 0.2


@dataclass(slots=True)
class DetectionResult:
	source_path: Path
	source_root: Path
	source_group: str
	image_width: int
	image_height: int
	score: float
	keep: bool
	xmin: int = -1
	ymin: int = -1
	xmax: int = -1
	ymax: int = -1
	preview_path: Path | None = None
	crop_path: Path | None = None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="screen 단일 클래스 자동 라벨링")
	parser.add_argument("--input-roots", type=Path, nargs="*", default=DEFAULT_INPUT_ROOTS)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
	parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=("TRAIN", "VAL", "TEST"))
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--accept-threshold", type=float, default=0.68)
	parser.add_argument("--review-threshold", type=float, default=0.45)
	parser.add_argument("--recursive", action="store_true", default=True)
	parser.add_argument("--export-dataset", action="store_true")
	return parser.parse_args()


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
	return max(minimum, min(maximum, value))


def image_files_under(root: Path, recursive: bool = True) -> Iterable[Path]:
	if not root.exists():
		return []
	iterator = root.rglob("*") if recursive else root.iterdir()
	return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def collect_images(input_roots: list[Path], recursive: bool = True) -> list[tuple[Path, Path, str]]:
	items: list[tuple[Path, Path, str]] = []
	for root in input_roots:
		root = root.resolve()
		for image_path in image_files_under(root, recursive=recursive):
			relative_path = image_path.relative_to(root)
			source_group = relative_path.parts[0] if len(relative_path.parts) > 1 else image_path.stem
			items.append((image_path, root, source_group))
	return items


def dataset_image_name(source_root: Path, image_path: Path) -> str:
	relative_path = image_path.relative_to(source_root).as_posix()
	return f"{source_root.name}__{relative_path.replace('/', '__')}"


def preprocess_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
	edges = cv2.Canny(gray, 50, 150)
	kernel = np.ones((5, 5), np.uint8)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
	edges = cv2.dilate(edges, kernel, iterations=1)
	return gray, edges


def order_box_points(points: np.ndarray) -> np.ndarray:
	ordered = np.zeros((4, 2), dtype=np.float32)
	summed = points.sum(axis=1)
	diff = np.diff(points, axis=1)
	ordered[0] = points[np.argmin(summed)]
	ordered[2] = points[np.argmax(summed)]
	ordered[1] = points[np.argmin(diff)]
	ordered[3] = points[np.argmax(diff)]
	return ordered


def warp_box_region(gray: np.ndarray, box: np.ndarray) -> np.ndarray:
	ordered = order_box_points(box.astype(np.float32))
	width_a = np.linalg.norm(ordered[2] - ordered[3])
	width_b = np.linalg.norm(ordered[1] - ordered[0])
	height_a = np.linalg.norm(ordered[1] - ordered[2])
	height_b = np.linalg.norm(ordered[0] - ordered[3])
	warp_width = max(1, int(round(max(width_a, width_b))))
	warp_height = max(1, int(round(max(height_a, height_b))))
	destination = np.array(
		[
			[0, 0],
			[warp_width - 1, 0],
			[warp_width - 1, warp_height - 1],
			[0, warp_height - 1],
		],
		dtype=np.float32,
	)
	matrix = cv2.getPerspectiveTransform(ordered, destination)
	return cv2.warpPerspective(gray, matrix, (warp_width, warp_height))


def border_metrics(warped_gray: np.ndarray) -> tuple[float, float, float, float]:
	height, width = warped_gray.shape[:2]
	if height < 8 or width < 8:
		return 255.0, 0.0, 255.0, 0.0

	border_thickness = max(4, int(round(min(width, height) * 0.12)))
	border_thickness = min(border_thickness, max(2, min(width, height) // 3))
	inner_left = border_thickness
	inner_top = border_thickness
	inner_right = width - border_thickness
	inner_bottom = height - border_thickness
	if inner_right <= inner_left or inner_bottom <= inner_top:
		return 255.0, 0.0, 255.0, 0.0

	border_mask = np.zeros((height, width), dtype=bool)
	border_mask[:border_thickness, :] = True
	border_mask[-border_thickness:, :] = True
	border_mask[:, :border_thickness] = True
	border_mask[:, -border_thickness:] = True
	inner_mask = np.zeros((height, width), dtype=bool)
	inner_mask[inner_top:inner_bottom, inner_left:inner_right] = True

	border_pixels = warped_gray[border_mask]
	inner_pixels = warped_gray[inner_mask]
	if border_pixels.size == 0 or inner_pixels.size == 0:
		return 255.0, 0.0, 255.0, 0.0

	return (
		float(np.mean(border_pixels)),
		float(np.std(border_pixels)),
		float(np.mean(inner_pixels)),
		float(np.std(inner_pixels)),
	)


def border_black_ratio(warped_gray: np.ndarray) -> float:
	height, width = warped_gray.shape[:2]
	if height < 8 or width < 8:
		return 0.0

	border_thickness = max(4, int(round(min(width, height) * 0.12)))
	border_thickness = min(border_thickness, max(2, min(width, height) // 3))
	border_mask = np.zeros((height, width), dtype=bool)
	border_mask[:border_thickness, :] = True
	border_mask[-border_thickness:, :] = True
	border_mask[:, :border_thickness] = True
	border_mask[:, -border_thickness:] = True
	border_pixels = warped_gray[border_mask]
	if border_pixels.size == 0:
		return 0.0
	return float(np.mean(border_pixels <= 55))


def score_candidate(contour: np.ndarray, gray: np.ndarray, edges: np.ndarray, image_width: int, image_height: int) -> tuple[float, tuple[int, int, int, int], np.ndarray]:
	area = cv2.contourArea(contour)
	if area <= 0:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	image_area = float(image_width * image_height)
	area_ratio = area / image_area
	if area_ratio < MIN_SCREEN_AREA_RATIO or area_ratio > 0.98:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	perimeter = cv2.arcLength(contour, True)
	if perimeter <= 0:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
	hull = cv2.convexHull(contour)
	hull_area = cv2.contourArea(hull)
	solidity = area / hull_area if hull_area > 0 else 0.0

	rotated = cv2.minAreaRect(contour)
	(_, _), (rect_width, rect_height), _ = rotated
	if rect_width <= 0 or rect_height <= 0:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	rect_area = rect_width * rect_height
	fill_ratio = area / rect_area if rect_area > 0 else 0.0
	aspect_ratio = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
	aspect_score = 1.0 - clamp(abs(math.log(aspect_ratio)) / math.log(6.0), 0.0, 1.0)
	quad_score = 1.0 - clamp(abs(len(approx) - 4) / 4.0, 0.0, 1.0)
	fill_score = clamp((fill_ratio - 0.45) / 0.5, 0.0, 1.0)
	solidity_score = clamp((solidity - 0.55) / 0.45, 0.0, 1.0)

	box = cv2.boxPoints(rotated)
	x_min = max(0, int(np.floor(np.min(box[:, 0]))))
	y_min = max(0, int(np.floor(np.min(box[:, 1]))))
	x_max = min(image_width - 1, int(np.ceil(np.max(box[:, 0]))))
	y_max = min(image_height - 1, int(np.ceil(np.max(box[:, 1]))))
	if x_max <= x_min or y_max <= y_min:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	roi_gray = gray[y_min : y_max + 1, x_min : x_max + 1]
	roi_edges = edges[y_min : y_max + 1, x_min : x_max + 1]
	if roi_gray.size == 0:
		return 0.0, (-1, -1, -1, -1), np.empty((0, 2), dtype=np.int32)

	edge_score = clamp(float(np.mean(roi_edges > 0)) / 0.08, 0.0, 1.0)
	contrast_score = clamp(float(np.std(roi_gray)) / 35.0, 0.0, 1.0)
	glare_score = 1.0 - clamp(float(np.mean(roi_gray >= 245)) / 0.12, 0.0, 1.0)

	warped_gray = warp_box_region(gray, box)
	border_mean, border_std, inner_mean, inner_std = border_metrics(warped_gray)
	border_dark_score = clamp((180.0 - border_mean) / 180.0, 0.0, 1.0)
	border_consistency_score = 1.0 - clamp(border_std / 55.0, 0.0, 1.0)
	border_contrast_score = clamp((inner_mean - border_mean) / 80.0, 0.0, 1.0)
	inner_texture_score = clamp(inner_std / 45.0, 0.0, 1.0)
	border_black_score = clamp(border_black_ratio(warped_gray), 0.0, 1.0)
	size_score = clamp((area_ratio - MIN_SCREEN_AREA_RATIO) / 0.18, 0.0, 1.0)

	score = (
		0.16 * quad_score
		+ 0.08 * fill_score
		+ 0.08 * solidity_score
		+ 0.08 * aspect_score
		+ 0.05 * edge_score
		+ 0.04 * contrast_score
		+ 0.18 * border_dark_score
		+ 0.10 * border_consistency_score
		+ 0.10 * border_contrast_score
		+ 0.10 * inner_texture_score
		+ 0.13 * border_black_score
		+ 0.10 * size_score
	) * glare_score

	return score, (x_min, y_min, x_max, y_max), box.astype(np.int32)


def detect_screen_candidate(image_bgr: np.ndarray) -> tuple[float, tuple[int, int, int, int], np.ndarray]:
	image_height, image_width = image_bgr.shape[:2]
	gray, edges = preprocess_image(image_bgr)
	contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	best_score = 0.0
	best_box = (-1, -1, -1, -1)
	best_polygon = np.empty((0, 2), dtype=np.int32)
	for contour in sorted(contours, key=cv2.contourArea, reverse=True):
		score, box, polygon = score_candidate(contour, gray, edges, image_width, image_height)
		if score > best_score:
			best_score = score
			best_box = box
			best_polygon = polygon
	return best_score, best_box, best_polygon


def draw_preview(image_bgr: np.ndarray, box: tuple[int, int, int, int], polygon: np.ndarray, label: str) -> np.ndarray:
	preview = image_bgr.copy()
	x_min, y_min, x_max, y_max = box
	if x_min >= 0 and y_min >= 0 and x_max >= 0 and y_max >= 0:
		cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
	if polygon.size:
		cv2.polylines(preview, [polygon], True, (255, 0, 0), 2)
	cv2.putText(preview, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
	return preview


def scan_images(args: argparse.Namespace, run_dir: Path) -> list[DetectionResult]:
	review_dir = run_dir / "review"
	crop_dir = run_dir / "crops"
	review_dir.mkdir(parents=True, exist_ok=True)
	crop_dir.mkdir(parents=True, exist_ok=True)

	records: list[DetectionResult] = []
	for image_path, source_root, source_group in collect_images(args.input_roots, recursive=args.recursive):
		image_bgr = cv2.imread(str(image_path))
		if image_bgr is None:
			continue

		image_height, image_width = image_bgr.shape[:2]
		score, box, polygon = detect_screen_candidate(image_bgr)
		found = score >= args.review_threshold and box[0] >= 0
		keep = score >= args.accept_threshold and box[0] >= 0
		label = f"screen score={score:.3f} keep={int(keep)}" if found else f"no screen score={score:.3f}"

		preview_path = review_dir / f"{dataset_image_name(source_root, image_path)}"
		cv2.imwrite(str(preview_path), draw_preview(image_bgr, box, polygon, label))

		crop_path: Path | None = None
		if found:
			x_min, y_min, x_max, y_max = box
			crop = image_bgr[y_min : y_max + 1, x_min : x_max + 1]
			crop_path = crop_dir / f"{preview_path.stem}_crop{preview_path.suffix}"
			cv2.imwrite(str(crop_path), crop)

		records.append(
			DetectionResult(
				source_path=image_path,
				source_root=source_root,
				source_group=source_group,
				image_width=image_width,
				image_height=image_height,
				score=round(score, 4),
				keep=keep,
				xmin=box[0],
				ymin=box[1],
				xmax=box[2],
				ymax=box[3],
				preview_path=preview_path,
				crop_path=crop_path,
			)
		)

	return records


def record_to_row(record: DetectionResult) -> dict[str, object]:
	return {
		"source_path": str(record.source_path),
		"source_root": str(record.source_root),
		"source_group": record.source_group,
		"image_width": record.image_width,
		"image_height": record.image_height,
		"score": record.score,
		"keep": int(record.keep),
		"xmin": record.xmin,
		"ymin": record.ymin,
		"xmax": record.xmax,
		"ymax": record.ymax,
		"preview_path": str(record.preview_path) if record.preview_path else "",
		"crop_path": str(record.crop_path) if record.crop_path else "",
	}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
		if rows:
			writer.writeheader()
			writer.writerows(rows)


def write_manifest_csv(path: Path, records: list[DetectionResult]) -> None:
	write_csv(path, [record_to_row(record) for record in records])


def resolve_split_counts(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
	total = train_ratio + val_ratio + test_ratio
	if total <= 0:
		raise ValueError("split ratios must be positive")
	return train_ratio / total, val_ratio / total, test_ratio / total


def compute_split_sizes(total_items: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
	train_ratio, val_ratio, test_ratio = resolve_split_counts(train_ratio, val_ratio, test_ratio)
	ratios = [train_ratio, val_ratio, test_ratio]
	expected = [total_items * ratio for ratio in ratios]
	sizes = [int(math.floor(value)) for value in expected]
	remainders = [value - size for value, size in zip(expected, sizes)]

	positive_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]
	for index in positive_indices:
		if total_items > 0 and sizes[index] == 0:
			sizes[index] = 1

	while sum(sizes) > total_items:
		reducible_indices = [index for index in range(3) if sizes[index] > (1 if index in positive_indices else 0)]
		if not reducible_indices:
			break
		reducible_index = min(reducible_indices, key=lambda index: remainders[index])
		sizes[reducible_index] -= 1

	while sum(sizes) < total_items:
		addable_index = max(range(3), key=lambda index: remainders[index])
		sizes[addable_index] += 1
		remainders[addable_index] = 0.0

	return sizes[0], sizes[1], sizes[2]


def assign_splits(records: list[DetectionResult], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[Path, str]:
	ordered_records = list(records)
	random.Random(seed).shuffle(ordered_records)
	train_count, val_count, test_count = compute_split_sizes(len(ordered_records), train_ratio, val_ratio, test_ratio)
	train_cutoff = train_count
	val_cutoff = train_count + val_count
	split_map: dict[Path, str] = {}
	for index, record in enumerate(ordered_records):
		if index < train_cutoff:
			split_name = "train"
		elif index < val_cutoff:
			split_name = "val"
		else:
			split_name = "test"
		split_map[record.source_path] = split_name
	return split_map


def yolo_label_line(box: tuple[int, int, int, int], image_width: int, image_height: int) -> str:
	x_min, y_min, x_max, y_max = box
	x_center = ((x_min + x_max) / 2.0) / image_width
	y_center = ((y_min + y_max) / 2.0) / image_height
	width = (x_max - x_min) / image_width
	height = (y_max - y_min) / image_height
	return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def write_data_yaml(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	content = "\n".join(
		[
			f"path: {path.parent.as_posix()}",
			"train: images/train",
			"val: images/val",
			"test: images/test",
			"nc: 1",
			"names:",
			f"  0: {CLASS_NAME}",
		]
	) + "\n"
	path.write_text(content, encoding="utf-8")
	path.with_suffix(".yml").write_text(content, encoding="utf-8")


def export_dataset(args: argparse.Namespace, records: list[DetectionResult]) -> None:
	accepted = [record for record in records if record.keep and record.xmin >= 0]
	if not accepted:
		raise RuntimeError("내보낼 데이터가 없습니다. accept-threshold를 낮춰 보세요.")

	images_root = args.output_root / "images"
	labels_root = args.output_root / "labels"
	meta_root = args.output_root / "meta"
	if args.output_root.exists():
		for path in (images_root, labels_root, meta_root):
			if path.exists():
				shutil.rmtree(path)
		for path in (args.output_root / "data.yaml", args.output_root / "data.yml"):
			if path.exists():
				path.unlink()
	for split_name in ("train", "val", "test"):
		(images_root / split_name).mkdir(parents=True, exist_ok=True)
		(labels_root / split_name).mkdir(parents=True, exist_ok=True)
	meta_root.mkdir(parents=True, exist_ok=True)

	split_map = assign_splits(accepted, *args.split_ratios, seed=args.seed)
	exported_rows: list[dict[str, object]] = []
	for record in accepted:
		split_name = split_map.get(record.source_path, "train")
		dataset_name = dataset_image_name(record.source_root, record.source_path)
		destination_image = images_root / split_name / dataset_name
		destination_label = labels_root / split_name / destination_image.with_suffix(".txt").name
		shutil.copy2(record.source_path, destination_image)
		destination_label.write_text(yolo_label_line((record.xmin, record.ymin, record.xmax, record.ymax), record.image_width, record.image_height) + "\n", encoding="utf-8")
		exported_rows.append(
			{
				**record_to_row(record),
				"split": split_name,
				"dataset_image_path": str(destination_image),
				"dataset_label_path": str(destination_label),
				"class_id": CLASS_ID,
			}
		)

	write_csv(meta_root / "export_manifest.csv", exported_rows)
	write_data_yaml(args.output_root / "data.yaml")
	(meta_root / "run_summary.json").write_text(
		json.dumps(
			{
				"total_scanned": len(records),
				"accepted": len(accepted),
				"output_root": str(args.output_root),
				"split_ratios": {"train": args.split_ratios[0], "val": args.split_ratios[1], "test": args.split_ratios[2]},
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)


def main() -> None:
	args = parse_args()
	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = args.run_root / f"labeling_{run_ts}"
	run_dir.mkdir(parents=True, exist_ok=True)

	records = scan_images(args, run_dir)
	if not records:
		raise FileNotFoundError("입력 루트에서 이미지를 찾지 못했습니다.")

	write_manifest_csv(run_dir / "candidate_manifest.csv", records)
	summary = {
		"run_timestamp": run_ts,
		"total_scanned": len(records),
		"detected": len([record for record in records if record.xmin >= 0]),
		"accepted": len([record for record in records if record.keep]),
		"review_threshold": args.review_threshold,
		"accept_threshold": args.accept_threshold,
		"input_roots": [str(path) for path in args.input_roots],
		"run_dir": str(run_dir),
	}
	write_csv(run_dir / "run_summary.csv", [summary])
	(run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

	print(f"Saved review manifest: {run_dir / 'candidate_manifest.csv'}")
	print(f"Saved run summary: {run_dir / 'run_summary.json'}")

	if args.export_dataset:
		export_dataset(args, records)
		print(f"Saved dataset to: {args.output_root}")
		print(f"Saved data.yaml: {args.output_root / 'data.yaml'}")


if __name__ == "__main__":
	main()