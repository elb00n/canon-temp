"""Service layer package for orchestration and model access."""

from .target_service import (
	BackendName,
	TargetModelHandle,
	TargetService,
	load_target_model_handle,
	predict_target_bgr,
	resolve_target_openvino_model_path,
	resolve_target_weight_path,
)
from .target_test_service import TargetTestRunConfig, TargetTestService, TargetTestSummary
from .sequence_service import (
	SequenceRunConfig,
	SequenceService,
	TargetContext,
	TargetStepResult,
	VideoSequenceSummary,
)
from .video_service import (
	VIDEO_EXTENSIONS,
	VideoInfo,
	ensure_dir as ensure_video_dir,
	get_video_info,
	list_video_files as list_video_files_in_service,
	make_video_writer,
	open_video_capture,
	resolve_videos,
)

__all__ = [
	"BackendName",
	"TargetModelHandle",
	"TargetService",
	"TargetTestRunConfig",
	"TargetTestService",
	"TargetTestSummary",
	"SequenceRunConfig",
	"SequenceService",
	"TargetContext",
	"TargetStepResult",
	"VideoSequenceSummary",
	"VIDEO_EXTENSIONS",
	"VideoInfo",
	"ensure_video_dir",
	"get_video_info",
	"list_video_files_in_service",
	"make_video_writer",
	"open_video_capture",
	"resolve_videos",
	"load_target_model_handle",
	"predict_target_bgr",
	"resolve_target_openvino_model_path",
	"resolve_target_weight_path",
]
