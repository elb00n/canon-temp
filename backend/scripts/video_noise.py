"""Add blur and glare noise to videos without rotation or tilt changes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.core.paths import SAMPLE_VIDEO_DIR, VIDEO_NOISE_RUNS_DIR

DEFAULT_SOURCE_DIR = SAMPLE_VIDEO_DIR
DEFAULT_OUTPUT_DIR = VIDEO_NOISE_RUNS_DIR
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class NoiseConfig:
	preset: str = "fast"
	segment_seconds: float = 10.0
	blur_prob: float = 0.0
	glare_prob: float = 0.0
	blur_sigma_min: float = 0.5
	blur_sigma_max: float = 1.2
	glare_strength_min: float = 0.05
	glare_strength_max: float = 0.15
	glare_radius_ratio_min: float = 0.05
	glare_radius_ratio_max: float = 0.12
	brightness_delta_min: float = -3.0
	brightness_delta_max: float = 3.0
	seed: int | None = None


@dataclass(slots=True)
class SegmentNoiseState:
	mode: str
	brightness_delta: float
	blur_enabled: bool
	blur_sigma: float | None
	glare_enabled: bool
	glare_center_x: int | None
	glare_center_y: int | None
	glare_radius_x: int | None
	glare_radius_y: int | None
	glare_strength: float | None
	glare_mask: np.ndarray | None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Add blur and glare noise to videos")
	parser.add_argument("--source", type=Path, nargs="*", default=[DEFAULT_SOURCE_DIR], help="Source video files or folders")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder")
	parser.add_argument("--suffix", type=str, default="_noisy", help="Suffix added to output video filename")
	parser.add_argument("--preset", choices=["fast", "balanced", "strong"], default="fast", help="Noise strength preset")
	parser.add_argument("--blur-prob", type=float, default=0.0)
	parser.add_argument("--glare-prob", type=float, default=0.0)
	parser.add_argument("--blur-sigma-min", type=float, default=0.5)
	parser.add_argument("--blur-sigma-max", type=float, default=1.2)
	parser.add_argument("--glare-strength-min", type=float, default=0.05)
	parser.add_argument("--glare-strength-max", type=float, default=0.15)
	parser.add_argument("--glare-radius-ratio-min", type=float, default=0.05)
	parser.add_argument("--glare-radius-ratio-max", type=float, default=0.12)
	parser.add_argument("--brightness-delta-min", type=float, default=-3.0)
	parser.add_argument("--brightness-delta-max", type=float, default=3.0)
	parser.add_argument("--segment-seconds", type=float, default=10.0, help="Keep the same noise pattern for this many seconds")
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos")
	return parser.parse_args()


def list_video_files(root: Path) -> Iterable[Path]:
	if root.is_file():
		if root.suffix.lower() in VIDEO_EXTENSIONS:
			yield root
		return
	if not root.exists():
		return
	for path in sorted(root.rglob("*")):
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
			yield path


def resolve_sources(sources: list[Path]) -> list[Path]:
	resolved: list[Path] = []
	for source in sources:
		resolved.extend(list_video_files(source))
	return list(dict.fromkeys(sorted(resolved)))


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def clamp_uint8(image_bgr: np.ndarray) -> np.ndarray:
	return np.clip(image_bgr, 0, 255).astype(np.uint8)


def apply_blur(frame_bgr: np.ndarray, rng: np.random.Generator, config: NoiseConfig) -> np.ndarray:
	sigma = float(rng.uniform(config.blur_sigma_min, config.blur_sigma_max))
	kernel_size = max(3, int(round(sigma * 4)) | 1)
	return cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), sigmaX=sigma)


def apply_glare(frame_bgr: np.ndarray, rng: np.random.Generator, config: NoiseConfig) -> np.ndarray:
	height, width = frame_bgr.shape[:2]
	center_x = int(rng.integers(width // 8, max(width // 8 + 1, width - width // 8)))
	center_y = int(rng.integers(height // 8, max(height // 8 + 1, height - height // 8)))
	radius_ratio = float(rng.uniform(config.glare_radius_ratio_min, config.glare_radius_ratio_max))
	radius_x = max(8, int(width * radius_ratio))
	radius_y = max(8, int(height * radius_ratio * 0.8))
	strength = float(rng.uniform(config.glare_strength_min, config.glare_strength_max))

	mask = np.zeros((height, width), dtype=np.float32)
	cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1.0, -1)
	mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius_x, radius_y) * 0.35)
	mask = np.clip(mask, 0.0, 1.0)[..., None]

	glare_color = np.full_like(frame_bgr, 255, dtype=np.float32)
	frame_float = frame_bgr.astype(np.float32)
	blended = frame_float * (1.0 - mask * strength) + glare_color * (mask * strength)
	blended = cv2.GaussianBlur(blended, (0, 0), sigmaX=max(radius_x, radius_y) * 0.18)
	return clamp_uint8(blended)


def adjust_brightness(frame_bgr: np.ndarray, rng: np.random.Generator, config: NoiseConfig) -> np.ndarray:
	delta = float(rng.uniform(config.brightness_delta_min, config.brightness_delta_max))
	frame_float = frame_bgr.astype(np.float32) + delta
	return clamp_uint8(frame_float)


def sample_segment_state(rng: np.random.Generator, config: NoiseConfig, frame_shape: tuple[int, int, int]) -> SegmentNoiseState:
	height, width = frame_shape[:2]
	available_modes: list[str] = ["brightness"]
	if config.blur_prob > 0.0:
		available_modes.append("blur")
	if config.glare_prob > 0.0:
		available_modes.append("glare")
	if config.preset == "fast" or len(available_modes) == 1:
		mode = "brightness"
	else:
		mode = str(rng.choice(available_modes))
	brightness_delta = float(rng.uniform(config.brightness_delta_min, config.brightness_delta_max)) if mode == "brightness" else 0.0
	blur_enabled = mode == "blur" and bool(rng.random() < config.blur_prob)
	blur_sigma = float(rng.uniform(config.blur_sigma_min, config.blur_sigma_max)) if blur_enabled else None
	glare_enabled = mode == "glare" and bool(rng.random() < config.glare_prob)
	if glare_enabled:
		center_x = int(rng.integers(width // 8, max(width // 8 + 1, width - width // 8)))
		center_y = int(rng.integers(height // 8, max(height // 8 + 1, height - height // 8)))
		radius_ratio = float(rng.uniform(config.glare_radius_ratio_min, config.glare_radius_ratio_max))
		radius_x = max(8, int(width * radius_ratio))
		radius_y = max(8, int(height * radius_ratio * 0.8))
		glare_strength = float(rng.uniform(config.glare_strength_min, config.glare_strength_max))
	else:
		center_x = None
		center_y = None
		radius_x = None
		radius_y = None
		glare_strength = None
	glare_mask = None
	if glare_enabled and center_x is not None and center_y is not None and radius_x is not None and radius_y is not None:
		mask = np.zeros((height, width), dtype=np.float32)
		cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1.0, -1)
		mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(radius_x, radius_y) * 0.35)
		glare_mask = np.clip(mask, 0.0, 1.0)[..., None]
	return SegmentNoiseState(
		mode=mode,
		brightness_delta=brightness_delta,
		blur_enabled=blur_enabled,
		blur_sigma=blur_sigma,
		glare_enabled=glare_enabled,
		glare_center_x=center_x,
		glare_center_y=center_y,
		glare_radius_x=radius_x,
		glare_radius_y=radius_y,
		glare_strength=glare_strength,
		glare_mask=glare_mask,
	)


def add_noise_to_frame(frame_bgr: np.ndarray, state: SegmentNoiseState) -> np.ndarray:
	result = frame_bgr.copy()
	if state.mode == "brightness":
		result = cv2.convertScaleAbs(result, alpha=1.0, beta=state.brightness_delta)
	if state.blur_enabled and state.blur_sigma is not None:
		kernel_size = max(3, int(round(state.blur_sigma * 4)) | 1)
		result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigmaX=state.blur_sigma)
	if state.glare_enabled and state.glare_mask is not None:
		height, width = result.shape[:2]
		mask = state.glare_mask
		radius = (max(8, int(state.glare_radius_x or width * 0.1)), max(8, int(state.glare_radius_y or height * 0.08)))
		glare_strength = float(state.glare_strength or 0.0)
		glare_color = np.full_like(result, 255, dtype=np.float32)
		frame_float = result.astype(np.float32)
		blended = frame_float * (1.0 - mask * glare_strength) + glare_color * (mask * glare_strength)
		blended = cv2.GaussianBlur(blended, (0, 0), sigmaX=max(radius) * 0.18)
		result = clamp_uint8(blended)
	return result


def process_video(video_path: Path, output_dir: Path, suffix: str, config: NoiseConfig, overwrite: bool) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{video_path.stem}{suffix}{video_path.suffix}"
	if output_path.exists() and not overwrite:
		return output_path

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"failed to open video: {video_path}")

	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	if fps <= 0.0:
		fps = 30.0
	segment_frames = max(1, int(round(fps * max(config.segment_seconds, 0.1))))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	if width <= 0 or height <= 0:
		raise RuntimeError(f"failed to read video size: {video_path}")

	fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path.suffix.lower() == ".mp4" else cv2.VideoWriter_fourcc(*"XVID")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
	if not writer.isOpened():
		cap.release()
		raise RuntimeError(f"failed to open writer: {output_path}")

	rng = np.random.default_rng(config.seed)
	frame_count = 0
	current_segment_index = -1
	current_state: SegmentNoiseState | None = None
	try:
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				break
			segment_index = frame_count // segment_frames
			if current_state is None or segment_index != current_segment_index:
				current_state = sample_segment_state(rng, config, frame_bgr.shape)
				current_segment_index = segment_index
			noisy_frame = add_noise_to_frame(frame_bgr, current_state)
			writer.write(noisy_frame)
			frame_count += 1
	finally:
		cap.release()
		writer.release()

	print(f"Processed {video_path.name} -> {output_path} ({frame_count} frames)")
	return output_path


def main() -> None:
	args = parse_args()
	sources = resolve_sources(args.source)
	if not sources:
		raise FileNotFoundError("No video files found under the provided sources")

	config = NoiseConfig(
		preset=args.preset,
		segment_seconds=args.segment_seconds,
		blur_prob=args.blur_prob,
		glare_prob=args.glare_prob,
		blur_sigma_min=args.blur_sigma_min,
		blur_sigma_max=args.blur_sigma_max,
		glare_strength_min=args.glare_strength_min,
		glare_strength_max=args.glare_strength_max,
		glare_radius_ratio_min=args.glare_radius_ratio_min,
		glare_radius_ratio_max=args.glare_radius_ratio_max,
		brightness_delta_min=args.brightness_delta_min,
		brightness_delta_max=args.brightness_delta_max,
		seed=args.seed,
	)

	for source in sources:
		process_video(source, args.output_dir, args.suffix, config, args.overwrite)


if __name__ == "__main__":
	main()
