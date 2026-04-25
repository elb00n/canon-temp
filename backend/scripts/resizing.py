"""Resize target images into a cached target_images folder.

This script keeps the original target_image folder intact and writes padded,
resized copies into images/target_images for downstream matching.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.core.paths import TARGET_IMAGE_DIR, TARGET_IMAGES_DIR

DEFAULT_SOURCE_DIR = TARGET_IMAGE_DIR
DEFAULT_OUTPUT_DIR = TARGET_IMAGES_DIR
DEFAULT_OUTPUT_SIZE = 640
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class ResizeSummary:
	run_timestamp: str
	source_dir: str
	output_dir: str
	output_size: int
	images_seen: int
	images_saved: int
	cleared_existing: bool


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Resize target images into a cached folder")
	parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="원본 target 이미지 폴더")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="리사이즈된 target 이미지 저장 폴더")
	parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="패딩 후 리사이즈할 크기")
	parser.add_argument("--fill-value", type=int, default=0, help="패딩 영역 채울 값")
	parser.add_argument("--keep-existing", action="store_true", help="기존 output-dir 내용을 삭제하지 않음")
	return parser.parse_args()


def list_image_files(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


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


def resize_target_image(image_bgr: np.ndarray, output_size: int, fill_value: int) -> np.ndarray:
	letterboxed = pad_to_square(image_bgr, fill_value=fill_value)
	return cv2.resize(letterboxed, (output_size, output_size), interpolation=cv2.INTER_AREA)


def clear_directory(directory: Path) -> bool:
	if not directory.exists():
		return False
	cleared = False
	for path in list(directory.iterdir()):
		cleared = True
		if path.is_dir():
			shutil.rmtree(path)
		else:
			path.unlink()
	return cleared


def run_resize(source_dir: Path, output_dir: Path, output_size: int, fill_value: int, keep_existing: bool) -> ResizeSummary:
	if not source_dir.exists():
		raise FileNotFoundError(f"source target folder not found: {source_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	cleared_existing = False
	if not keep_existing:
		cleared_existing = clear_directory(output_dir)

	images_seen = 0
	images_saved = 0
	for source_path in list_image_files(source_dir):
		images_seen += 1
		image_bgr = load_bgr_image(source_path)
		resized = resize_target_image(image_bgr, output_size=output_size, fill_value=fill_value)
		destination_path = output_dir / source_path.name
		if cv2.imwrite(str(destination_path), resized):
			images_saved += 1
		else:
			raise RuntimeError(f"failed to write image: {destination_path}")

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	summary = ResizeSummary(
		run_timestamp=run_ts,
		source_dir=str(source_dir),
		output_dir=str(output_dir),
		output_size=output_size,
		images_seen=images_seen,
		images_saved=images_saved,
		cleared_existing=cleared_existing,
	)
	(output_dir / "resize_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
	return summary


def main() -> None:
	args = parse_args()
	summary = run_resize(
		args.source_dir,
		args.output_dir,
		args.output_size,
		args.fill_value,
		args.keep_existing,
	)
	print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
