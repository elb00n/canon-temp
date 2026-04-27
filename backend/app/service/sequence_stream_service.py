"""Streaming sequence detection service for live camera or RTSP sources."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import SETTINGS
from app.core.paths import ASSET_WEIGHTS_DIR, SEQUENCE_STREAM_RUNS_DIR, yolo_weight_file
from app.models.target_model import DEFAULT_THRESHOLD, get_target_sequence_settings, get_target_threshold
from app.models.warping import YoloScreenWarper
from db.database import DEFAULT_DB_PATH, upsert_sequence_run
from app.service.sequence_service import FrameRecord, SequenceService, TargetContext, TargetStepResult, VideoSequenceSummary, append_summary_csv, ensure_dir
from app.service.target_service import TargetModelHandle, TargetService


def resolve_capture_source(source: str | int | Path) -> int | str:
	if isinstance(source, int):
		return source
	if isinstance(source, Path):
		return str(source)
	text = str(source).strip()
	if text.isdigit():
		return int(text)
	return text


@dataclass(slots=True)
class StreamRunConfig:
	source: str | int | Path
	source_label: str = "stream"
	target_order: list[str] | None = None
	target_root: Path = ASSET_WEIGHTS_DIR
	yolo_weights: Path = yolo_weight_file()
	output_root: Path = SEQUENCE_STREAM_RUNS_DIR
	db_path: Path = DEFAULT_DB_PATH
	device: str = SETTINGS.sequence.device
	threshold: float | None = None
	conf: float = 0.25
	imgsz: int = SETTINGS.sequence.input_size
	padding_ratio: float = 0.02
	frame_step: int = 1
	sample_seconds: float = 0.5
	min_consecutive: int = SETTINGS.sequence.min_consecutive
	max_missed: int = 4
	save_confirmed_frames: bool = False
	confirmed_pre_roll: int = 3
	show_preview: bool = True
	window_name: str = "Sequence Stream"
	prefer_openvino: bool = True


@dataclass(slots=True)
class StreamFrameResult:
	frame_index: int
	timestamp_seconds: float
	detections_seen: int
	prediction_label: str | None
	prediction_score: float | None
	current_target: str
	consecutive_yes: int
	completed_target: bool


class LiveFrameReader:
	"""Continuously read the latest frame from a live source."""

	def __init__(self, source: int | str) -> None:
		self.source = source
		self.capture = cv2.VideoCapture(source)
		if not self.capture.isOpened():
			raise RuntimeError(f"failed to open stream source: {source}")
		self.lock = threading.Lock()
		self.thread: threading.Thread | None = None
		self.running = False
		self.ended = False
		self.latest_frame: np.ndarray | None = None
		self.latest_frame_index = 0
		self.latest_capture_time = 0.0

	def start(self) -> None:
		if self.running:
			return
		self.running = True
		self.thread = threading.Thread(target=self._run, daemon=True)
		self.thread.start()

	def _run(self) -> None:
		while self.running:
			ok, frame = self.capture.read()
			if not ok:
				self.ended = True
				break
			with self.lock:
				self.latest_frame = frame
				self.latest_frame_index += 1
				self.latest_capture_time = time.monotonic()
		self.running = False

	def read_latest(self, last_seen_index: int) -> tuple[int, np.ndarray, float] | None:
		with self.lock:
			if self.latest_frame is None or self.latest_frame_index <= last_seen_index:
				return None
			return self.latest_frame_index, self.latest_frame.copy(), self.latest_capture_time

	def stop(self) -> None:
		self.running = False
		if self.thread is not None:
			self.thread.join(timeout=1.0)
		self.capture.release()


class SequenceStreamService:
	"""Run the sequential target detector on a live camera or RTSP stream."""

	def __init__(self, config: StreamRunConfig) -> None:
		self.config = config
		self.target_service = TargetService(
			target_root=config.target_root,
			device=config.device,
			prefer_openvino=config.prefer_openvino,
		)

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

	def _build_warper(self) -> YoloScreenWarper:
		return YoloScreenWarper(
			weights=self.config.yolo_weights,
			device=self.config.device,
			conf=self.config.conf,
			imgsz=self.config.imgsz,
			padding_ratio=self.config.padding_ratio,
			output_size=640,
			classes=[0],
		)

	def _save_confirmed_frames(
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
			preview_bgr = SequenceService.draw_preview(
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
		confirmed_preview = SequenceService.draw_confirmed_preview(
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

	def process_stream(self) -> dict[str, object]:
		if not self.config.yolo_weights.exists():
			raise FileNotFoundError(f"YOLO weights not found: {self.config.yolo_weights}")
		target_order = list(self.config.target_order or ["target_1", "target_2", "target_3", "target_4"])
		if not target_order:
			raise ValueError("target-order must contain at least one target")

		run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		run_root = self.config.output_root / f"run_{run_ts}"
		ensure_dir(run_root)
		stream_dir = run_root / self.config.source_label
		confirmed_dir = stream_dir / "confirmed_frames"
		preview_dir = stream_dir / "preview"
		ensure_dir(stream_dir)
		ensure_dir(preview_dir)
		if self.config.save_confirmed_frames:
			ensure_dir(confirmed_dir)

		warper = self._build_warper()
		capture_source = resolve_capture_source(self.config.source)
		reader = LiveFrameReader(capture_source)
		reader.start()

		current_context = self.resolve_target_context(target_order[0])
		current_result = TargetStepResult(
			target_name=current_context.target_name,
			weights=str(current_context.handle.weights),
			threshold=current_context.threshold,
		)
		target_results: list[TargetStepResult] = [current_result]
		backends_used: set[str] = {current_context.handle.backend}
		frame_buffer: deque[FrameRecord] = deque(maxlen=max(0, int(self.config.confirmed_pre_roll)))

		processed_frames = 0
		total_detections = 0
		confirmed_frames_saved = 0
		consecutive_yes = 0
		missed_frames = 0
		completed = False
		frame_index = 0
		last_seen_frame_index = 0
		last_processed_capture_time = 0.0
		last_preview = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)

		try:
			while True:
				latest = reader.read_latest(last_seen_frame_index)
				if latest is None:
					if reader.ended:
						break
					if self.config.show_preview:
						cv2.imshow(self.config.window_name, last_preview)
						key = cv2.waitKey(1) & 0xFF
						if key == ord("q"):
							break
					time.sleep(0.005)
					continue

				last_seen_frame_index, frame_bgr, capture_time = latest
				if capture_time - last_processed_capture_time < current_context.sample_seconds:
					if self.config.show_preview:
						cv2.imshow(self.config.window_name, last_preview)
						key = cv2.waitKey(1) & 0xFF
						if key == ord("q"):
							break
					continue

				if self.config.frame_step > 1 and processed_frames > 0 and (last_seen_frame_index - frame_index) < self.config.frame_step:
					continue

				last_processed_capture_time = capture_time
				frame_index = last_seen_frame_index
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

				preview = SequenceService.draw_preview(
					frame_bgr,
					detections,
					target_name=current_context.target_name,
					threshold=current_context.threshold,
					prediction_label=prediction_label,
					prediction_score=prediction_score,
					consecutive_yes=consecutive_yes,
					min_consecutive=current_context.min_consecutive,
				)
				last_preview = preview

				if consecutive_yes >= current_context.min_consecutive:
					current_result.completed = True
					current_result.confirmed_frame = frame_index
					if self.config.save_confirmed_frames and prediction_label is not None and prediction_score is not None:
						confirmed_frames_saved += self._save_confirmed_frames(
							confirmed_dir,
							current_context.target_name,
							current_context.threshold,
							current_context.min_consecutive,
							frame_index,
							frame_buffer,
						)
					target_index = target_order.index(current_context.target_name) + 1
					consecutive_yes = 0
					missed_frames = 0
					frame_buffer.clear()
					if target_index >= len(target_order):
						completed = True
						break

					current_context = self.resolve_target_context(target_order[target_index])
					backends_used.add(current_context.handle.backend)
					current_result = TargetStepResult(
						target_name=current_context.target_name,
						weights=str(current_context.handle.weights),
						threshold=current_context.threshold,
					)
					target_results.append(current_result)

				if self.config.show_preview:
					cv2.imshow(self.config.window_name, preview)
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"):
						break

		finally:
			reader.stop()
			if self.config.show_preview:
				cv2.destroyAllWindows()

		return self._build_result(
			run_ts=run_ts,
			run_root=run_root,
			stream_dir=stream_dir,
			completed=completed,
			processed_frames=processed_frames,
			total_detections=total_detections,
			target_order=target_order,
			target_results=target_results,
			backends_used=backends_used,
			confirmed_frames_saved=confirmed_frames_saved,
		)

	def _build_result(
		self,
		*,
		run_ts: str,
		run_root: Path,
		stream_dir: Path,
		completed: bool,
		processed_frames: int,
		total_detections: int,
		target_order: list[str],
		target_results: list[TargetStepResult],
		backends_used: set[str],
		confirmed_frames_saved: int,
	) -> dict[str, object]:
		confirmed_targets = sum(1 for result in target_results if result.completed)
		stream_summary = VideoSequenceSummary(
			video_name=self.config.source_label,
			video_path=str(self.config.source),
			output_dir=str(stream_dir),
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
		(stream_dir / "stream_summary.json").write_text(
			json.dumps(asdict(stream_summary), indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		append_summary_csv(run_root / "run_summary.csv", {
			"run_timestamp": run_ts,
			"source": str(self.config.source),
			"completed": completed,
			"processed_frames": processed_frames,
			"confirmed_targets": confirmed_targets,
			"target_order": ",".join(target_order),
			"sample_seconds": self.config.sample_seconds,
			"min_consecutive": self.config.min_consecutive,
			"max_missed": self.config.max_missed,
			"device": self.config.device,
			"backends_used": ",".join(sorted(backends_used)),
		})
		json_path = run_root / f"run_{run_ts}.json"
		json_path.write_text(
			json.dumps(
				{
					"run_timestamp": run_ts,
					"config": asdict(stream_summary),
					"videos": [asdict(stream_summary)],
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
				"summary": {
					"run_timestamp": run_ts,
					"source": str(self.config.source),
					"completed": completed,
					"processed_frames": processed_frames,
					"confirmed_targets": confirmed_targets,
					"target_order": ",".join(target_order),
					"sample_seconds": self.config.sample_seconds,
					"min_consecutive": self.config.min_consecutive,
					"max_missed": self.config.max_missed,
					"device": self.config.device,
					"backends_used": ",".join(sorted(backends_used)),
				},
				"videos": [asdict(stream_summary)],
			},
			db_path=self.config.db_path,
		)
		return {
			"run_timestamp": run_ts,
			"run_root": str(run_root),
			"summary": asdict(stream_summary),
			"videos": [asdict(stream_summary)],
			"json_path": str(json_path),
			"db_path": str(self.config.db_path),
			"db_run_id": db_run_id,
		}


__all__ = [
	"LiveFrameReader",
	"SequenceStreamService",
	"StreamFrameResult",
	"StreamRunConfig",
	"resolve_capture_source",
]