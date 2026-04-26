"""Centralized project paths and shared defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSET_WEIGHTS_DIR = ASSETS_DIR / "weight"
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"

TARGET_IMAGE_DIR = IMAGES_DIR / "target_image"
TARGET_IMAGES_DIR = IMAGES_DIR / "target_images"
CNN_TRAIN_DIR = IMAGES_DIR / "cnn_train"
SAMPLE_IMAGE_DIR = IMAGES_DIR / "sample_images"
SAMPLE_IMAGES_FROM_VIDEOS_DIR = IMAGES_DIR / "sample_images_from_videos"
SAMPLE_VIDEO_DIR = IMAGES_DIR / "sample_video"

CNN_OUTPUT_DIR = OUTPUTS_DIR / "cnn_train"
YOLO_RUNS_DIR = OUTPUTS_DIR / "yolo_runs"
TARGET_TEST_RUNS_DIR = OUTPUTS_DIR / "target_test_runs"
SEQUENCE_VIDEO_RUNS_DIR = OUTPUTS_DIR / "sequence_video_runs"
SEQUENCE_STREAM_RUNS_DIR = OUTPUTS_DIR / "sequence_stream_runs"
VIDEO_NOISE_RUNS_DIR = OUTPUTS_DIR / "video_noise_runs"
WARPING_TEST_RUNS_DIR = OUTPUTS_DIR / "warping_test_runs"
TEMPLATE_MATCHER_RUNS_DIR = OUTPUTS_DIR / "template_matcher_v1"

DEFAULT_TARGET_MODEL_ROOT = ASSET_WEIGHTS_DIR
DEFAULT_YOLO_WEIGHTS = ASSET_WEIGHTS_DIR / "yolo" / "best.pt"
DEFAULT_ASSET_TARGET_MODEL_ROOT = ASSET_WEIGHTS_DIR
DEFAULT_ASSET_YOLO_WEIGHTS = ASSET_WEIGHTS_DIR / "yolo" / "best.pt"


def project_path(*parts: str | Path) -> Path:
	"""Build a path from the project root."""
	return PROJECT_ROOT.joinpath(*parts)


def ensure_directories(*paths: Path) -> None:
	"""Create one or more directories if they do not exist."""
	for path in paths:
		path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class TrainingDefaults:
	input_size: int = 640
	batch_size: int = 8
	epochs: int = 10
	lr: float = 1e-4
	weight_decay: float = 1e-4
	val_ratio: float = 0.2
	seed: int = 42
	early_stop_patience: int = 4
	early_stop_min_delta: float = 1e-4
	device: str = "cpu"


@dataclass(frozen=True, slots=True)
class SequenceDefaults:
	threshold: float = 0.5
	min_consecutive: int = 3
	sample_fps: float = 6.0
	timeout_seconds: float = 30.0
	input_size: int = 640
	device: str = "cpu"
	openvino_device: str = "CPU"


@dataclass(frozen=True, slots=True)
class NoiseDefaults:
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


@dataclass(frozen=True, slots=True)
class AppSettings:
	project_root: Path = PROJECT_ROOT
	app_dir: Path = APP_DIR
	scripts_dir: Path = SCRIPTS_DIR
	images_dir: Path = IMAGES_DIR
	outputs_dir: Path = OUTPUTS_DIR
	weights_dir: Path = WEIGHTS_DIR
	assets_dir: Path = ASSETS_DIR
	asset_weights_dir: Path = ASSET_WEIGHTS_DIR
	data_dir: Path = DATA_DIR
	db_dir: Path = DB_DIR
	configs_dir: Path = CONFIGS_DIR
	tests_dir: Path = TESTS_DIR

	target_image_dir: Path = TARGET_IMAGE_DIR
	target_images_dir: Path = TARGET_IMAGES_DIR
	cnn_train_dir: Path = CNN_TRAIN_DIR
	sample_image_dir: Path = SAMPLE_IMAGE_DIR
	sample_images_from_videos_dir: Path = SAMPLE_IMAGES_FROM_VIDEOS_DIR
	sample_video_dir: Path = SAMPLE_VIDEO_DIR

	cnn_output_dir: Path = CNN_OUTPUT_DIR
	yolo_runs_dir: Path = YOLO_RUNS_DIR
	target_test_runs_dir: Path = TARGET_TEST_RUNS_DIR
	sequence_video_runs_dir: Path = SEQUENCE_VIDEO_RUNS_DIR
	video_noise_runs_dir: Path = VIDEO_NOISE_RUNS_DIR
	sequence_stream_runs_dir: Path = SEQUENCE_STREAM_RUNS_DIR
	warping_test_runs_dir: Path = WARPING_TEST_RUNS_DIR
	template_matcher_runs_dir: Path = TEMPLATE_MATCHER_RUNS_DIR

	default_target_model_root: Path = DEFAULT_TARGET_MODEL_ROOT
	default_yolo_weights: Path = DEFAULT_YOLO_WEIGHTS
	default_asset_target_model_root: Path = DEFAULT_ASSET_TARGET_MODEL_ROOT
	default_asset_yolo_weights: Path = DEFAULT_ASSET_YOLO_WEIGHTS
	training: TrainingDefaults = TrainingDefaults()
	sequence: SequenceDefaults = SequenceDefaults()
	noise: NoiseDefaults = NoiseDefaults()


SETTINGS = AppSettings()
