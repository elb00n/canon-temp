"""Video I/O helpers used by service-layer orchestration code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class VideoInfo:
	path: Path
	fps: float
	frame_width: int
	frame_height: int
	frame_count: int


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def list_video_files(root: Path) -> list[Path]:
	if not root.exists():
		return []
	if root.is_file():
		return [root] if root.suffix.lower() in VIDEO_EXTENSIONS else []
	return sorted(
		path
		for path in root.rglob("*")
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	)


def resolve_videos(sources: list[Path]) -> list[Path]:
	videos: list[Path] = []
	for source in sources:
		videos.extend(list_video_files(source))
	return list(dict.fromkeys(sorted(videos)))


def get_video_info(video_path: Path) -> VideoInfo:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"failed to open video: {video_path}")
	try:
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		return VideoInfo(
			path=video_path,
			fps=fps if fps > 0.0 else 30.0,
			frame_width=frame_width,
			frame_height=frame_height,
			frame_count=frame_count,
		)
	finally:
		cap.release()


def open_video_capture(video_path: Path) -> cv2.VideoCapture:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"failed to open video: {video_path}")
	return cap


def make_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
	fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path.suffix.lower() == ".mp4" else cv2.VideoWriter_fourcc(*"XVID")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"failed to open writer: {output_path}")
	return writer


__all__ = [
	"VIDEO_EXTENSIONS",
	"VideoInfo",
	"ensure_dir",
	"get_video_info",
	"list_video_files",
	"make_video_writer",
	"open_video_capture",
	"resolve_videos",
]
