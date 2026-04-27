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
from .sequence_stream_service import SequenceStreamService, StreamRunConfig
from .sequence_db import DEFAULT_DB_PATH, StoredSequenceRun, connect as connect_sequence_db, initialize as initialize_sequence_db, upsert_sequence_run
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
from .classifier_adapter import MockTargetClassifier, MockTargetClassifierSuite, RealTargetClassifier, RealTargetClassifierSuite
from .decision_engine import DecisionConfig, DecisionEngine
from .detector_adapter import MockScreenDetector, ScreenDetectorAdapter, YoloDetectorAdapter
from .operational_db import DEFAULT_OPERATIONAL_DB_PATH, OperationalLogStore
from .operational_pipeline import OperationalInferenceService, OperationalPipelineConfig, build_synthetic_frame
from .preprocess_service import ScreenPreprocessor
from .state_service import SequenceStateMachine, SequenceStateRegistry

__all__ = [
	"BackendName",
	"TargetModelHandle",
	"TargetService",
	"TargetTestRunConfig",
	"TargetTestService",
	"TargetTestSummary",
	"SequenceRunConfig",
	"SequenceService",
	"SequenceStreamService",
	"StreamRunConfig",
	"TargetContext",
	"TargetStepResult",
	"VideoSequenceSummary",
	"DEFAULT_DB_PATH",
	"StoredSequenceRun",
	"connect_sequence_db",
	"initialize_sequence_db",
	"upsert_sequence_run",
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
	"DEFAULT_OPERATIONAL_DB_PATH",
	"DecisionConfig",
	"DecisionEngine",
	"MockScreenDetector",
	"MockTargetClassifier",
	"MockTargetClassifierSuite",
	"OperationalInferenceService",
	"OperationalLogStore",
	"OperationalPipelineConfig",
	"RealTargetClassifier",
	"RealTargetClassifierSuite",
	"ScreenDetectorAdapter",
	"ScreenPreprocessor",
	"SequenceStateMachine",
	"SequenceStateRegistry",
	"YoloDetectorAdapter",
	"build_synthetic_frame",
]
