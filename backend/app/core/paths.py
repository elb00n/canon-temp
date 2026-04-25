"""Reusable project path helpers built on top of centralized settings."""

from __future__ import annotations

from pathlib import Path

from app.core.config import SETTINGS, project_path


PROJECT_ROOT = SETTINGS.project_root
APP_DIR = SETTINGS.app_dir
SCRIPTS_DIR = SETTINGS.scripts_dir
IMAGES_DIR = SETTINGS.images_dir
OUTPUTS_DIR = SETTINGS.outputs_dir
WEIGHTS_DIR = SETTINGS.weights_dir
ASSETS_DIR = SETTINGS.assets_dir
ASSET_WEIGHTS_DIR = SETTINGS.asset_weights_dir

TARGET_IMAGE_DIR = SETTINGS.target_image_dir
TARGET_IMAGES_DIR = SETTINGS.target_images_dir
CNN_TRAIN_DIR = SETTINGS.cnn_train_dir
SAMPLE_IMAGE_DIR = SETTINGS.sample_image_dir
SAMPLE_IMAGES_FROM_VIDEOS_DIR = SETTINGS.sample_images_from_videos_dir
SAMPLE_VIDEO_DIR = SETTINGS.sample_video_dir

CNN_OUTPUT_DIR = SETTINGS.cnn_output_dir
YOLO_RUNS_DIR = SETTINGS.yolo_runs_dir
TARGET_TEST_RUNS_DIR = SETTINGS.target_test_runs_dir
SEQUENCE_VIDEO_RUNS_DIR = SETTINGS.sequence_video_runs_dir
VIDEO_NOISE_RUNS_DIR = SETTINGS.video_noise_runs_dir
WARPING_TEST_RUNS_DIR = SETTINGS.warping_test_runs_dir
TEMPLATE_MATCHER_RUNS_DIR = SETTINGS.template_matcher_runs_dir


def asset_weight_dir(target_name: str) -> Path:
	"""Return the directory that stores the canonical target weights."""
	return ASSET_WEIGHTS_DIR / target_name


def asset_weight_file(target_name: str) -> Path:
	"""Return the canonical PyTorch weight path for a target."""
	return asset_weight_dir(target_name) / "best.pt"


def asset_openvino_dir(target_name: str) -> Path:
	"""Return the OpenVINO output directory for a target."""
	return asset_weight_dir(target_name) / f"{target_name}_weight_openvino"


def asset_openvino_model_file(target_name: str) -> Path:
	"""Return the OpenVINO XML path for a target if it exists."""
	return asset_openvino_dir(target_name) / "model.xml"


def yolo_weight_file(filename: str = "yolo11n.pt") -> Path:
	"""Return a YOLO weight file stored under assets/weight/yolo."""
	return ASSET_WEIGHTS_DIR / "yolo" / filename


def target_test_source_dir() -> Path:
	"""Return the default source folder for target test runs."""
	return OUTPUTS_DIR / "warped"


def ensure_project_dirs() -> None:
	"""Create the core runtime directories if they are missing."""
	for path in (
		IMAGES_DIR,
		OUTPUTS_DIR,
		ASSETS_DIR,
		ASSET_WEIGHTS_DIR,
		TARGET_IMAGE_DIR,
		TARGET_IMAGES_DIR,
		CNN_TRAIN_DIR,
		SAMPLE_IMAGE_DIR,
		SAMPLE_IMAGES_FROM_VIDEOS_DIR,
		SAMPLE_VIDEO_DIR,
		CNN_OUTPUT_DIR,
		YOLO_RUNS_DIR,
		TARGET_TEST_RUNS_DIR,
		SEQUENCE_VIDEO_RUNS_DIR,
		VIDEO_NOISE_RUNS_DIR,
		WARPING_TEST_RUNS_DIR,
		TEMPLATE_MATCHER_RUNS_DIR,
	):
		path.mkdir(parents=True, exist_ok=True)


__all__ = [
	"PROJECT_ROOT",
	"APP_DIR",
	"SCRIPTS_DIR",
	"IMAGES_DIR",
	"OUTPUTS_DIR",
	"WEIGHTS_DIR",
	"ASSETS_DIR",
	"ASSET_WEIGHTS_DIR",
	"TARGET_IMAGE_DIR",
	"TARGET_IMAGES_DIR",
	"CNN_TRAIN_DIR",
	"SAMPLE_IMAGE_DIR",
	"SAMPLE_IMAGES_FROM_VIDEOS_DIR",
	"SAMPLE_VIDEO_DIR",
	"CNN_OUTPUT_DIR",
	"YOLO_RUNS_DIR",
	"TARGET_TEST_RUNS_DIR",
	"SEQUENCE_VIDEO_RUNS_DIR",
	"VIDEO_NOISE_RUNS_DIR",
	"WARPING_TEST_RUNS_DIR",
	"TEMPLATE_MATCHER_RUNS_DIR",
	"asset_weight_dir",
	"asset_weight_file",
	"asset_openvino_dir",
	"asset_openvino_model_file",
	"yolo_weight_file",
	"target_test_source_dir",
	"ensure_project_dirs",
	"project_path",
	"SETTINGS",
]
