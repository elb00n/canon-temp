"""Sequential video detection service.

This module coordinates video frame sampling, YOLO detection, warping,
target classification, and run summary persistence.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import SETTINGS
from app.core.paths import ASSET_WEIGHTS_DIR, SAMPLE_VIDEO_DIR, SEQUENCE_VIDEO_RUNS_DIR, yolo_weight_file
from app.models.target_model import DEFAULT_THRESHOLD, BinaryPrediction, get_target_sequence_settings, get_target_threshold
from app.models.warping import YoloScreenWarper
from db.database import DEFAULT_DB_PATH, upsert_sequence_run
from app.service.target_service import TargetModelHandle, TargetService
from app.service.video_service import ensure_dir, open_video_capture, resolve_videos


@dataclass(slots=True)
class SequenceRunConfig:
	source: list[Path]
	target_order: list[str]
	target_root: Path = ASSET_WEIGHTS_DIR
	yolo_weights: Path = yolo_weight_file()
	output_root: Path = SEQUENCE_VIDEO_RUNS_DIR
	db_path: Path = DEFAULT_DB_PATH
	device: str = SETTINGS.sequence.device
	threshold: float | None = None
	conf: float = 0.25
	imgsz: int = SETTINGS.sequence.input_size
	padding_ratio: float = 0.02
	frame_step: float = 1.0
	sample_seconds: float = 0.5
	min_consecutive: int = SETTINGS.sequence.min_consecutive
	max_missed: int = 4
	save_confirmed_frames: bool = False
	confirmed_pre_roll: int = 3


@dataclass(slots=True)
class TargetStepResult:
	target_name: str
	weights: str
	threshold: float
	start_frame: int | None = None
	confirmed_frame: int | None = None
	completed: bool = False
	processed_frames: int = 0
	detections_seen: int = 0
	yes_frames: int = 0
	no_frames: int = 0
	no_detection_frames: int = 0
	consecutive_yes_max: int = 0
	last_label: str = ""
	last_score: float = 0.0
	last_prob_yes: float = 0.0
	last_prob_no: float = 0.0


@dataclass(slots=True)
class VideoSequenceSummary:
	video_name: str
	video_path: str
	output_dir: str
	completed: bool
	confirmed_targets: int
	processed_frames: int
	total_detections: int
	total_yes_frames: int
	total_no_frames: int
	total_no_detection_frames: int
	min_consecutive: int
	max_missed: int
	frame_step: int
	device: str
	confidence_threshold: float
	image_size: int
	padding_ratio: float
	sample_seconds: float
	target_order: list[str]
	targets: list[TargetStepResult]
	backends_used: list[str]
	confirmed_frames_saved: int


@dataclass(slots=True)
class FrameRecord:
	frame_index: int
	frame_bgr: np.ndarray
	detections: list[dict[str, object]]
	prediction_label: str | None
	prediction_score: float | None


@dataclass(slots=True)
class TargetContext:
	target_name: str
	handle: TargetModelHandle
	threshold: float
	min_consecutive: int
	sample_seconds: float
	max_missed: int


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def append_summary_csv(csv_path: Path, row: dict[str, object]) -> None:
	ensure_dir(csv_path.parent)
	write_header = not csv_path.exists()
	with csv_path.open("a", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
		if write_header:
			writer.writeheader()
		writer.writerow(row)


class SequenceService:
	"""Run sequential target detection over videos."""

	def __init__(self, config: SequenceRunConfig) -> None:
		self.config = config
		self.target_service = TargetService(target_root=config.target_root, device=config.device, prefer_openvino=True)

	def resolve_target_context(self, target_name: str) -> TargetContext:
		threshold = self.config.threshold if self.config.threshold is not None else get_target_threshold(target_name, DEFAULT_THRESHOLD)
		sequence_settings = get_target_sequence_settings(target_name)
		handle = self.target_service.get_handle(target_name)
		return TargetContext(
			target_name=target_name,
			handle=handle,
			threshold=threshold,
			min_consecutive=int(sequence_settings.get("min_consecutive", self.config.min_consecutive)),
			sample_seconds=float(sequence_settings.get("sample_seconds", self.config.sample_seconds)),
			max_missed=int(sequence_settings.get("max_missed", self.config.max_missed)),
		)

	@staticmethod
	def draw_preview(
		frame_bgr: np.ndarray,
		detections: list[dict[str, object]],
		*,
		target_name: str,
		threshold: float,
		prediction_label: str | None = None,
		prediction_score: float | None = None,
		consecutive_yes: int = 0,
		min_consecutive: int = 0,
	) -> np.ndarray:
		preview = frame_bgr.copy()
		for detection in detections:
			bbox_xyxy = detection["bbox_xyxy"]
			x_min, y_min, x_max, y_max = [int(round(value)) for value in bbox_xyxy]
			label = f'{detection["class_name"]} {float(detection["confidence"]):.2f}'
			cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
			label_y = max(18, y_min - 8)
			cv2.putText(preview, label, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

		header = f"{target_name} thr={threshold:.2f} yes={consecutive_yes}/{min_consecutive}"
		if prediction_label is not None and prediction_score is not None:
			header += f" pred={prediction_label}:{prediction_score:.3f}"
		cv2.putText(preview, header, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
		return preview

	@classmethod
	def draw_confirmed_preview(
		cls,
		frame_bgr: np.ndarray,
		detections: list[dict[str, object]],
		*,
		target_name: str,
		threshold: float,
		prediction_label: str,
		prediction_score: float,
		consecutive_yes: int,
		min_consecutive: int,
		frame_index: int,
	) -> np.ndarray:
		preview = cls.draw_preview(
			frame_bgr,
			detections,
			target_name=target_name,
			threshold=threshold,
			prediction_label=prediction_label,
			prediction_score=prediction_score,
			consecutive_yes=consecutive_yes,
			min_consecutive=min_consecutive,
		)
		cv2.putText(preview, f"CONFIRMED frame={frame_index}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
		return preview

	def save_confirmed_frames(
		self,
		confirmed_dir: Path,
		current_target: str,
		threshold: float,
		min_consecutive: int,
		confirmed_frame_index: int,
		frame_buffer: deque[FrameRecord],
	) -> int:
		saved_count = 0
		for offset, record in enumerate(frame_buffer, start=1):
			preview_bgr = self.draw_preview(
				record.frame_bgr,
				record.detections,
				target_name=current_target,
				threshold=threshold,
				prediction_label=record.prediction_label,
				prediction_score=record.prediction_score,
				consecutive_yes=max(0, offset - 1),
				min_consecutive=min_consecutive,
			)
			cv2.putText(
				preview_bgr,
				f"PREROLL frame={record.frame_index}",
				(18, 68),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 200, 0),
				2,
				cv2.LINE_AA,
			)
			confirmed_path = confirmed_dir / f"{current_target}_frame_{record.frame_index:06d}_pre.jpg"
			cv2.imwrite(str(confirmed_path), preview_bgr)
			saved_count += 1

		confirmed_path = confirmed_dir / f"{current_target}_frame_{confirmed_frame_index:06d}_confirmed.jpg"
		confirmed_preview = self.draw_confirmed_preview(
			frame_buffer[-1].frame_bgr if frame_buffer else np.zeros((640, 640, 3), dtype=np.uint8),
			frame_buffer[-1].detections if frame_buffer else [],
			target_name=current_target,
			threshold=threshold,
			prediction_label=frame_buffer[-1].prediction_label if frame_buffer else "yes",
			prediction_score=frame_buffer[-1].prediction_score if frame_buffer else 1.0,
			consecutive_yes=min_consecutive,
			min_consecutive=min_consecutive,
			frame_index=confirmed_frame_index,
		)
		cv2.imwrite(str(confirmed_path), confirmed_preview)
		return saved_count + 1

	def process_video(self, video_path: Path, run_root: Path) -> dict[str, object]:
		output_dir = run_root / video_path.stem
		confirmed_dir = output_dir / "confirmed_frames"
		ensure_dir(output_dir)
		if self.config.save_confirmed_frames:
			ensure_dir(confirmed_dir)

		warper = YoloScreenWarper(
			weights=self.config.yolo_weights,
			device=self.config.device,
			conf=self.config.conf,
			imgsz=self.config.imgsz,
			padding_ratio=self.config.padding_ratio,
			output_size=640,
			classes=[0],
		)

		cap = open_video_capture(video_path)

		video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		if video_fps <= 0.0:
			video_fps = 30.0

		target_order = list(self.config.target_order)
		if not target_order:
			raise ValueError("target-order must contain at least one target")

		target_index = 0
		current_context = self.resolve_target_context(target_order[target_index])
		effective_frame_step = max(1, int(round(video_fps * max(current_context.sample_seconds, 0.0))))
		if self.config.frame_step > 1:
			effective_frame_step = max(effective_frame_step, int(self.config.frame_step))
		current_result = TargetStepResult(
			target_name=current_context.target_name,
			weights=str(current_context.handle.weights),
			threshold=current_context.threshold,
		)
		target_results: list[TargetStepResult] = [current_result]
		backends_used: set[str] = {current_context.handle.backend}

		frame_index = 0
		processed_frames = 0
		total_detections = 0
		confirmed_frames_saved = 0
		consecutive_yes = 0
		missed_frames = 0
		completed = False
		frame_buffer: deque[FrameRecord] = deque(maxlen=max(0, int(self.config.confirmed_pre_roll)))

		while True:
			if effective_frame_step > 1:
				for _ in range(effective_frame_step - 1):
					if not cap.grab():
						cap.release()
						return self._build_video_result(
							video_path=video_path,
							output_dir=output_dir,
							completed=completed,
							processed_frames=processed_frames,
							total_detections=total_detections,
							target_order=target_order,
							target_results=target_results,
							backends_used=backends_used,
							confirmed_frames_saved=confirmed_frames_saved,
						)
			ok, frame_bgr = cap.retrieve()
			if not ok:
				break

			frame_index += 1
			processed_frames += 1
			if current_result.start_frame is None:
				current_result.start_frame = frame_index
			current_result.processed_frames += 1

			detections = warper.detect(frame_bgr)
			total_detections += len(detections)
			current_result.detections_seen += len(detections)

			prediction_label: str | None = None
			prediction_score: float | None = None

			if not detections:
				current_result.no_detection_frames += 1
				missed_frames += 1
				if missed_frames > current_context.max_missed:
					consecutive_yes = 0
			else:
				missed_frames = 0
				best_detection = max(detections, key=lambda detection: detection["confidence"])
				warped_detection = warper.warp_detection(frame_bgr, best_detection, index=0)
				prediction = current_context.handle.predict_bgr(
					warped_detection.warped_bgr,
					device=self.config.device,
					threshold=current_context.threshold,
				)
				prediction_label = prediction.label
				prediction_score = prediction.score
				current_result.last_label = prediction.label
				current_result.last_score = prediction.score
				current_result.last_prob_yes = prediction.prob_yes
				current_result.last_prob_no = prediction.prob_no
				if prediction.label == "yes":
					current_result.yes_frames += 1
					consecutive_yes += 1
					current_result.consecutive_yes_max = max(current_result.consecutive_yes_max, consecutive_yes)
				else:
					current_result.no_frames += 1
					consecutive_yes = 0

			frame_buffer.append(
				FrameRecord(
					frame_index=frame_index,
					frame_bgr=frame_bgr.copy(),
					detections=[dict(detection) for detection in detections],
					prediction_label=prediction_label,
					prediction_score=prediction_score,
				),
			)

			if consecutive_yes >= current_context.min_consecutive:
				current_result.completed = True
				current_result.confirmed_frame = frame_index
				if self.config.save_confirmed_frames and prediction_label is not None and prediction_score is not None:
					confirmed_frames_saved += self.save_confirmed_frames(
						confirmed_dir,
						current_context.target_name,
						current_context.threshold,
						current_context.min_consecutive,
						frame_index,
						frame_buffer,
					)
				target_index += 1
				consecutive_yes = 0
				missed_frames = 0
				frame_buffer.clear()
				if target_index >= len(target_order):
					completed = True
					break

				current_context = self.resolve_target_context(target_order[target_index])
				backends_used.add(current_context.handle.backend)
				effective_frame_step = max(1, int(round(video_fps * max(current_context.sample_seconds, 0.0))))
				if self.config.frame_step > 1:
					effective_frame_step = max(effective_frame_step, int(self.config.frame_step))
				current_result = TargetStepResult(
					target_name=current_context.target_name,
					weights=str(current_context.handle.weights),
					threshold=current_context.threshold,
				)
				target_results.append(current_result)

			if effective_frame_step > 1 and not completed:
				for _ in range(effective_frame_step - 1):
					if not cap.grab():
						break

		cap.release()
		return self._build_video_result(
			video_path=video_path,
			output_dir=output_dir,
			completed=completed,
			processed_frames=processed_frames,
			total_detections=total_detections,
			target_order=target_order,
			target_results=target_results,
			backends_used=backends_used,
			confirmed_frames_saved=confirmed_frames_saved,
		)

	def _build_video_result(
		self,
		*,
		video_path: Path,
		output_dir: Path,
		completed: bool,
		processed_frames: int,
		total_detections: int,
		target_order: list[str],
		target_results: list[TargetStepResult],
		backends_used: set[str],
		confirmed_frames_saved: int,
	) -> dict[str, object]:
		confirmed_targets = sum(1 for result in target_results if result.completed)
		video_summary = VideoSequenceSummary(
			video_name=video_path.name,
			video_path=str(video_path),
			output_dir=str(output_dir),
			completed=completed,
			confirmed_targets=confirmed_targets,
			processed_frames=processed_frames,
			total_detections=total_detections,
			total_yes_frames=sum(result.yes_frames for result in target_results),
			total_no_frames=sum(result.no_frames for result in target_results),
			total_no_detection_frames=sum(result.no_detection_frames for result in target_results),
			min_consecutive=self.config.min_consecutive,
			max_missed=self.config.max_missed,
			frame_step=self.config.frame_step,
			device=self.config.device,
			confidence_threshold=self.config.threshold if self.config.threshold is not None else DEFAULT_THRESHOLD,
			image_size=self.config.imgsz,
			padding_ratio=self.config.padding_ratio,
			sample_seconds=self.config.sample_seconds,
			target_order=target_order,
			targets=target_results,
			confirmed_frames_saved=confirmed_frames_saved,
			backends_used=sorted(backends_used),
		)
		(output_dir / "video_summary.json").write_text(
			json.dumps(asdict(video_summary), indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		return {
			**asdict(video_summary),
			"confirmed_frames_saved": confirmed_frames_saved,
			"backends_used": sorted(backends_used),
		}

	def run(self) -> dict[str, object]:
		for source in self.config.source:
			if not source.exists():
				raise FileNotFoundError(f"source not found: {source}")
		if not self.config.yolo_weights.exists():
			raise FileNotFoundError(f"YOLO weights not found: {self.config.yolo_weights}")

		videos = resolve_videos(self.config.source)
		if not videos:
			raise FileNotFoundError("No video files found under the provided sources")

		run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		run_root = self.config.output_root / f"run_{run_ts}"
		ensure_dir(run_root)

		video_results: list[dict[str, object]] = []
		for video_path in videos:
			result = self.process_video(video_path, run_root)
			video_results.append(result)

		total_videos = len(video_results)
		completed_videos = sum(1 for result in video_results if bool(result["completed"]))
		total_frames = sum(int(result["processed_frames"]) for result in video_results)
		total_confirmed_targets = sum(int(result["confirmed_targets"]) for result in video_results)
		backends_used = sorted({backend for result in video_results for backend in result.get("backends_used", [])})

		summary_row = {
			"run_timestamp": run_ts,
			"total_videos": total_videos,
			"completed_videos": completed_videos,
			"completion_rate": round(completed_videos / total_videos, 4),
			"total_processed_frames": total_frames,
			"total_confirmed_targets": total_confirmed_targets,
			"target_order": ",".join(self.config.target_order),
			"threshold_override": self.config.threshold if self.config.threshold is not None else "per-target",
			"min_consecutive": self.config.min_consecutive,
			"max_missed": self.config.max_missed,
			"frame_step": self.config.frame_step,
			"sample_seconds": self.config.sample_seconds,
			"device": self.config.device,
			"backends_used": ",".join(backends_used),
			"confidence_threshold": self.config.conf,
			"image_size": self.config.imgsz,
			"padding_ratio": self.config.padding_ratio,
		}

		append_summary_csv(run_root / "run_summary.csv", summary_row)

		json_path = run_root / f"run_{run_ts}.json"
		json_path.write_text(
			json.dumps(
				{
					"run_timestamp": run_ts,
					"config": summary_row,
					"videos": video_results,
				},
				indent=2,
				ensure_ascii=False,
			),
			encoding="utf-8",
		)
		db_run_id = upsert_sequence_run(
			{
				"run_timestamp": run_ts,
				"run_root": str(run_root),
				"summary": summary_row,
				"videos": video_results,
			},
			db_path=self.config.db_path,
		)
		return {
			"run_timestamp": run_ts,
			"run_root": str(run_root),
			"summary": summary_row,
			"videos": video_results,
			"json_path": str(json_path),
			"db_path": str(self.config.db_path),
			"db_run_id": db_run_id,
		}


__all__ = [
	"FrameRecord",
	"SequenceRunConfig",
	"SequenceService",
	"TargetContext",
	"TargetStepResult",
	"VideoSequenceSummary",
	"append_summary_csv",
	"ensure_dir",
	"list_video_files",
	"resolve_videos",
]
