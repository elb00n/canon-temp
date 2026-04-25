from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_IMAGES_DIR = BASE_DIR / "images" / "target_images"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_MIN_GOOD_MATCHES = 8


def list_image_files(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_bgr_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"failed to read image: {image_path}")
	return image


def target_name_from_path(image_path: Path, targets_dir: Path) -> str:
	relative = image_path.relative_to(targets_dir)
	if len(relative.parts) > 1:
		return relative.parts[0]
	return image_path.stem


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
	if image_bgr.ndim == 2:
		return image_bgr
	return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


@dataclass(slots=True)
class SIFTTargetItem:
	target_name: str
	image_path: Path
	keypoints: list[cv2.KeyPoint]
	descriptors: np.ndarray | None


@dataclass(slots=True)
class SIFTMatchScore:
	target_name: str
	image_path: str
	good_matches: int
	raw_matches: int
	inliers: int
	inlier_ratio: float
	score: float


class SIFTExtractor:
	def __init__(self, *, nfeatures: int = 2000) -> None:
		if not hasattr(cv2, "SIFT_create"):
			raise RuntimeError("OpenCV SIFT is not available. Install opencv-contrib-python.")
		self.extractor = cv2.SIFT_create(nfeatures=nfeatures)

	def extract(self, image_bgr: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
		gray = to_gray(image_bgr)
		return self.extractor.detectAndCompute(gray, None)


class SIFTTargetCache:
	def __init__(self, targets_dir: Path = DEFAULT_TARGET_IMAGES_DIR, *, nfeatures: int = 2000) -> None:
		self.targets_dir = targets_dir
		self.extractor = SIFTExtractor(nfeatures=nfeatures)
		self.items = self._build_cache()

	def _build_cache(self) -> list[SIFTTargetItem]:
		items: list[SIFTTargetItem] = []
		for image_path in list_image_files(self.targets_dir):
			image = load_bgr_image(image_path)
			keypoints, descriptors = self.extractor.extract(image)
			items.append(
				SIFTTargetItem(
					target_name=target_name_from_path(image_path, self.targets_dir),
					image_path=image_path,
					keypoints=keypoints,
					descriptors=descriptors,
				)
			)
		if not items:
			raise RuntimeError(f"No target images found in {self.targets_dir}")
		return items

	def group_by_target(self) -> dict[str, list[SIFTTargetItem]]:
		groups: dict[str, list[SIFTTargetItem]] = {}
		for item in self.items:
			groups.setdefault(item.target_name, []).append(item)
		return groups


def count_inliers(query_keypoints: list[cv2.KeyPoint], target_keypoints: list[cv2.KeyPoint], query_matches: list[list[cv2.DMatch]]) -> tuple[int, float]:
	if len(query_matches) < 4:
		return 0, 0.0

	query_points = []
	target_points = []
	for pair in query_matches:
		if len(pair) != 2:
			continue
		best, second = pair
		query_points.append(query_keypoints[best.queryIdx].pt)
		target_points.append(target_keypoints[best.trainIdx].pt)

	if len(query_points) < 4:
		return 0, 0.0

	query_array = np.float32(query_points).reshape(-1, 1, 2)
	target_array = np.float32(target_points).reshape(-1, 1, 2)
	_, mask = cv2.findHomography(query_array, target_array, cv2.RANSAC, 5.0)
	if mask is None:
		return 0, 0.0
	inliers = int(mask.ravel().sum())
	ratio = inliers / max(1, len(query_points))
	return inliers, ratio


class SIFTMatcher:
	def __init__(self, targets_dir: Path = DEFAULT_TARGET_IMAGES_DIR, *, ratio_threshold: float = DEFAULT_RATIO_THRESHOLD, min_good_matches: int = DEFAULT_MIN_GOOD_MATCHES, nfeatures: int = 2000) -> None:
		self.cache = SIFTTargetCache(targets_dir, nfeatures=nfeatures)
		self.extractor = self.cache.extractor
		self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		self.ratio_threshold = ratio_threshold
		self.min_good_matches = min_good_matches

	def _ratio_test(self, query_descriptors: np.ndarray | None, target_descriptors: np.ndarray | None) -> list[list[cv2.DMatch]]:
		if query_descriptors is None or target_descriptors is None:
			return []
		if len(query_descriptors) == 0 or len(target_descriptors) == 0:
			return []
		matches = self.matcher.knnMatch(query_descriptors, target_descriptors, k=2)
		good: list[list[cv2.DMatch]] = []
		for pair in matches:
			if len(pair) < 2:
				continue
			best, second = pair
			if best.distance < self.ratio_threshold * second.distance:
				good.append([best, second])
		return good

	def score(self, query_bgr: np.ndarray, target_names: set[str] | None = None) -> list[SIFTMatchScore]:
		query_keypoints, query_descriptors = self.extractor.extract(query_bgr)
		scores: list[SIFTMatchScore] = []
		for item in self.cache.items:
			if target_names is not None and item.target_name not in target_names:
				continue
			good_matches = self._ratio_test(query_descriptors, item.descriptors)
			good_count = len(good_matches)
			inliers, inlier_ratio = count_inliers(query_keypoints, item.keypoints, good_matches)
			raw_matches = 0 if query_descriptors is None or item.descriptors is None else len(query_descriptors)
			score = float(good_count + inliers * 2 + inlier_ratio * 10)
			scores.append(
				SIFTMatchScore(
					target_name=item.target_name,
					image_path=str(item.image_path),
					good_matches=good_count,
					raw_matches=raw_matches,
					inliers=inliers,
					inlier_ratio=round(inlier_ratio, 4),
					score=round(score, 4),
				)
			)
		scores.sort(key=lambda item: item.score, reverse=True)
		return scores

	def match(self, query_bgr: np.ndarray) -> SIFTMatchScore | None:
		scores = self.score(query_bgr)
		if not scores:
			return None
		best = scores[0]
		if best.good_matches < self.min_good_matches:
			return best
		return best
