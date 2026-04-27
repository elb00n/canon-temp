"""Real crop, optional deskew, resize, and normalization for inference."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import SETTINGS
from app.models.warping import crop_with_padding, flatten_screen, warp_screen_from_crop
from app.service.operational_types import PreprocessResult, PreprocessVariant


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_bgr_for_resnet(image_bgr: np.ndarray) -> np.ndarray:
	"""Return CHW float32 ImageNet-normalized data for classifier adapters."""
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	chw = image_rgb.transpose(2, 0, 1)
	return ((chw - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


class ScreenPreprocessor:
	"""Keep the production preprocessing path active for mock and real models."""

	def __init__(
		self,
		*,
		contour_min_area_ratio: float = 0.70,
		contour_target_aspect_ratio: float = 1.80,
		contour_aspect_ratio_tolerance: float = 0.45,
	) -> None:
		self.contour_min_area_ratio = contour_min_area_ratio
		self.contour_target_aspect_ratio = contour_target_aspect_ratio
		self.contour_aspect_ratio_tolerance = contour_aspect_ratio_tolerance

	def preprocess(
		self,
		image_bgr: np.ndarray,
		bbox: tuple[float, float, float, float],
		variant: PreprocessVariant,
	) -> PreprocessResult:
		bbox_array = np.asarray(bbox, dtype=np.float32)
		crop_bgr = crop_with_padding(image_bgr, bbox_array, variant.padding_ratio)
		deskew_attempted = bool(variant.deskew)
		deskew_applied = False
		working_bgr = crop_bgr

		if variant.deskew:
			warped_bgr = warp_screen_from_crop(
				crop_bgr,
				min_area_ratio=self.contour_min_area_ratio,
				target_aspect_ratio=self.contour_target_aspect_ratio,
				aspect_ratio_tolerance=self.contour_aspect_ratio_tolerance,
			)
			if warped_bgr is not None:
				working_bgr = warped_bgr
				deskew_applied = True

		processed_bgr = flatten_screen(working_bgr, variant.output_size)
		normalized_chw = normalize_bgr_for_resnet(processed_bgr)
		return PreprocessResult(
			variant=variant,
			bbox=tuple(float(value) for value in bbox_array),
			crop_bgr=crop_bgr,
			processed_bgr=processed_bgr,
			normalized_chw=normalized_chw,
			deskew_attempted=deskew_attempted,
			deskew_applied=deskew_applied,
		)


def default_preprocess_variant() -> PreprocessVariant:
	return PreprocessVariant(
		name="base",
		padding_ratio=SETTINGS.operational.PREPROCESS_PADDING_RATIO,
		deskew=True,
		output_size=SETTINGS.operational.PREPROCESS_OUTPUT_SIZE,
	)


def reinspect_preprocess_variants() -> list[PreprocessVariant]:
	base_size = SETTINGS.operational.PREPROCESS_OUTPUT_SIZE
	return [
		default_preprocess_variant(),
		PreprocessVariant(name="expanded_crop", padding_ratio=0.08, deskew=True, output_size=base_size),
		PreprocessVariant(name="deskew_off", padding_ratio=SETTINGS.operational.PREPROCESS_PADDING_RATIO, deskew=False, output_size=base_size),
		PreprocessVariant(name="resize_variant", padding_ratio=SETTINGS.operational.PREPROCESS_PADDING_RATIO, deskew=True, output_size=672),
	]


__all__ = [
	"ScreenPreprocessor",
	"default_preprocess_variant",
	"normalize_bgr_for_resnet",
	"reinspect_preprocess_variants",
]
