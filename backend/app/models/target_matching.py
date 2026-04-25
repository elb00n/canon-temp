from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from app.models.sift import SIFTMatcher


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS_DIR = BASE_DIR / "images" / "target_image"
DEFAULT_TARGET_IMAGES_DIR = BASE_DIR / "images" / "target_images"
DEFAULT_WARPED_DIR = BASE_DIR / "images" / "warped_640"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "target_matching"
DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_OUTPUT_SIZE = 640
DEFAULT_SIFT_TOP_K = 3
DEFAULT_SIFT_WEIGHT = 0.01
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Match warped images to target images with ResNet18 features")
	parser.add_argument("--query", type=Path, default=None, help="비교할 단일 warped 이미지")
	parser.add_argument("--warped-dir", type=Path, default=DEFAULT_WARPED_DIR, help="비교할 warped 이미지 폴더")
	parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS_DIR, help="기준 target 이미지 폴더")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE, help="피처 추출 전 리사이즈 크기")
	parser.add_argument("--sift-top-k", type=int, default=DEFAULT_SIFT_TOP_K, help="SIFT로 재검증할 ResNet 후보 수")
	parser.add_argument("--sift-weight", type=float, default=DEFAULT_SIFT_WEIGHT, help="SIFT 점수의 결합 가중치")
	return parser.parse_args()


def resolve_target_dir(targets_dir: Path) -> Path:
	if targets_dir.exists():
		return targets_dir
	raise FileNotFoundError(f"no target image folder found in {targets_dir}")


def ensure_target_images_dir(source_dir: Path, target_images_dir: Path, output_size: int) -> Path:
	target_images_dir.mkdir(parents=True, exist_ok=True)
	for existing_path in list(target_images_dir.iterdir()):
		if existing_path.is_dir():
			shutil.rmtree(existing_path)
		else:
			existing_path.unlink()

	for source_path in list_image_files(source_dir):
		image = load_bgr_image(source_path)
		padded = preprocess_for_embedding(image, output_size)
		destination_path = target_images_dir / source_path.name
		cv2.imwrite(str(destination_path), padded)

	return target_images_dir


def resolve_query_images(query_path: Path | None, warped_dir: Path) -> list[Path]:
	if query_path is not None:
		return [query_path]

	if not warped_dir.exists():
		raise FileNotFoundError(f"warped image folder not found: {warped_dir}")

	image_paths = list(list_image_files(warped_dir))
	if not image_paths:
		raise FileNotFoundError(f"no warped images found in {warped_dir}")
	return image_paths


def list_image_files(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def order_points(points: np.ndarray) -> np.ndarray:
	ordered = np.zeros((4, 2), dtype=np.float32)
	summed = points.sum(axis=1)
	diff = np.diff(points, axis=1)
	ordered[0] = points[np.argmin(summed)]
	ordered[2] = points[np.argmax(summed)]
	ordered[1] = points[np.argmin(diff)]
	ordered[3] = points[np.argmax(diff)]
	return ordered


def load_bgr_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"failed to read image: {image_path}")
	return image


def pad_to_square(image_bgr: np.ndarray, fill_value: int = 0) -> np.ndarray:
	if image_bgr.size == 0:
		return np.zeros((1, 1, 3), dtype=np.uint8)

	height, width = image_bgr.shape[:2]
	side = max(height, width)
	canvas = np.full((side, side, 3), fill_value, dtype=image_bgr.dtype)
	top = (side - height) // 2
	left = (side - width) // 2
	canvas[top : top + height, left : left + width] = image_bgr
	return canvas


def preprocess_for_embedding(crop_bgr: np.ndarray, output_size: int = DEFAULT_OUTPUT_SIZE) -> np.ndarray:
	if crop_bgr.size == 0:
		return np.zeros((output_size, output_size, 3), dtype=np.uint8)
	letterboxed = pad_to_square(crop_bgr)
	return cv2.resize(letterboxed, (output_size, output_size))


def target_name_from_path(image_path: Path, targets_dir: Path) -> str:
	relative = image_path.relative_to(targets_dir)
	if len(relative.parts) > 1:
		return str(relative.with_suffix(""))
	return image_path.stem


def image_name_from_path(image_path: Path, root_dir: Path) -> str:
	try:
		relative = image_path.relative_to(root_dir)
		return str(relative.with_suffix(""))
	except ValueError:
		return image_path.stem


def folder_name_from_target(target_name: str, is_match: bool) -> str:
	if not is_match:
		return "other"

	clean_name = target_name.strip().lower()
	if clean_name.startswith("target"):
		return clean_name
	if clean_name.isdigit():
		return f"target{clean_name}"
	return f"target_{clean_name}"


def sift_score_to_dict(score) -> dict[str, object] | None:
	if score is None:
		return None
	return {
		"target_name": score.target_name,
		"image_path": score.image_path,
		"good_matches": score.good_matches,
		"raw_matches": score.raw_matches,
		"inliers": score.inliers,
		"inlier_ratio": score.inlier_ratio,
		"score": score.score,
	}


def rerank_with_sift(
	resnet_scores: list[dict[str, object]],
	sift_matcher: SIFTMatcher,
	query_bgr: np.ndarray,
	top_k: int,
	sift_weight: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
	candidate_scores = resnet_scores[: max(1, top_k)]
	candidate_names = {str(item["target_name"]) for item in candidate_scores}
	sift_scores = sift_matcher.score(query_bgr, target_names=candidate_names)
	sift_by_name = {item.target_name: item for item in sift_scores}

	combined_scores: list[dict[str, object]] = []
	for candidate in candidate_scores:
		target_name = str(candidate["target_name"])
		resnet_score = float(candidate["score"])
		sift_score = sift_by_name.get(target_name)
		combined_score = resnet_score + (float(sift_score.score) * sift_weight if sift_score is not None else 0.0)
		combined_scores.append(
			{
				"target_name": target_name,
				"target_path": str(candidate["target_path"]),
				"resnet_score": resnet_score,
				"sift_score": sift_score_to_dict(sift_score),
				"combined_score": round(combined_score, 4),
			}
		)

	combined_scores.sort(key=lambda item: float(item["combined_score"]), reverse=True)
	best = combined_scores[0] if combined_scores else candidate_scores[0]
	serialized_sift_scores = []
	for item in sift_scores:
		serialized_item = sift_score_to_dict(item)
		if serialized_item is not None:
			serialized_sift_scores.append(serialized_item)
	return best, combined_scores, serialized_sift_scores


class FeatureExtractor:
	def __init__(self, device: str = "cpu") -> None:
		self.device = torch.device(device)
		weights = ResNet18_Weights.DEFAULT
		backbone = resnet18(weights=weights)
		self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
		self.model.eval()
		self.preprocess = weights.transforms()

	@torch.no_grad()
	def embed(self, image_bgr: np.ndarray) -> torch.Tensor:
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(image_rgb)
		tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
		feature = self.model(tensor).flatten(1)
		feature = F.normalize(feature, p=2, dim=1)
		return feature.squeeze(0).cpu()


class TargetMatcher:
	def __init__(self, targets_dir: Path, device: str = "cpu", output_size: int = DEFAULT_OUTPUT_SIZE) -> None:
		self.targets_dir = targets_dir
		self.output_size = output_size
		self.extractor = FeatureExtractor(device=device)
		self.targets = self._build_targets()

	def _build_targets(self) -> list[tuple[str, Path, torch.Tensor]]:
		targets: list[tuple[str, Path, torch.Tensor]] = []
		groups: dict[str, list[Path]] = {}
		image_paths = list(list_image_files(self.targets_dir))
		for image_path in image_paths:
			groups.setdefault(target_name_from_path(image_path, self.targets_dir), []).append(image_path)

		for target_name, paths in groups.items():
			for image_path in paths:
				image = load_bgr_image(image_path)
				embedding = self.extractor.embed(image)
				targets.append((target_name, image_path, embedding))
		if not targets:
			raise RuntimeError(f"No target images found in {self.targets_dir}")
		return targets

	def score_all(self, crop_bgr: np.ndarray) -> list[dict[str, object]]:
		flattened = preprocess_for_embedding(crop_bgr, self.output_size)
		embedding = self.extractor.embed(flattened)

		scores: list[dict[str, object]] = []
		for name, target_path, target_embedding in self.targets:
			score = float(F.cosine_similarity(embedding.unsqueeze(0), target_embedding.unsqueeze(0)).item())
			scores.append(
				{
					"target_name": name,
					"target_path": str(target_path),
					"score": score,
				}
			)
		scores.sort(key=lambda item: float(item["score"]), reverse=True)
		return scores

	def match(self, crop_bgr: np.ndarray) -> tuple[str, str, float, np.ndarray, list[dict[str, object]]]:
		flattened = preprocess_for_embedding(crop_bgr, self.output_size)
		scores = self.score_all(crop_bgr)

		best_name = "unknown"
		best_path = ""
		best_score = -1.0
		if scores:
			best = scores[0]
			best_name = str(best["target_name"])
			best_path = str(best["target_path"])
			best_score = float(best["score"])
		return best_name, best_path, best_score, flattened, scores


def match_query_images(
	matcher: TargetMatcher,
	sift_matcher: SIFTMatcher,
	query_paths: list[Path],
	output_dir: Path,
	threshold: float,
	sift_top_k: int,
	sift_weight: float,
	query_root: Path | None,
) -> list[dict[str, object]]:
	output_dir.mkdir(parents=True, exist_ok=True)

	results: list[dict[str, object]] = []
	for query_path in query_paths:
		if not query_path.exists():
			raise FileNotFoundError(f"query image not found: {query_path}")

		query_bgr = load_bgr_image(query_path)
		query_name = image_name_from_path(query_path, query_root) if query_root is not None else query_path.stem
		resnet_scores = matcher.score_all(query_bgr)
		best_candidate, combined_scores, sift_scores = rerank_with_sift(
			resnet_scores=resnet_scores,
			sift_matcher=sift_matcher,
			query_bgr=query_bgr,
			top_k=sift_top_k,
			sift_weight=sift_weight,
		)
		resnet_best = resnet_scores[0] if resnet_scores else {"target_name": "unknown", "target_path": "", "score": -1.0}
		best_sift = sift_scores[0] if sift_scores else None
		best_name = str(best_candidate["target_name"])
		best_path = str(best_candidate["target_path"])
		best_score = float(best_candidate.get("combined_score", best_candidate.get("resnet_score", -1.0)))
		flattened = preprocess_for_embedding(query_bgr, matcher.output_size)
		is_match = float(resnet_best["score"]) >= threshold

		folder_name = folder_name_from_target(best_name, is_match)
		destination_dir = output_dir / folder_name
		destination_dir.mkdir(parents=True, exist_ok=True)
		matched_name = f"{query_path.stem}_{best_name}_{best_score:.2f}.jpg"
		matched_path = destination_dir / matched_name
		cv2.imwrite(str(matched_path), flattened)

		results.append(
			{
				"query_name": query_name,
				"query_path": str(query_path),
				"target_name": best_name,
				"target_path": best_path,
				"score": best_score,
				"resnet_best": {
					"target_name": str(resnet_best["target_name"]),
					"target_path": str(resnet_best["target_path"]),
					"score": float(resnet_best["score"]),
				},
				"combined_scores": combined_scores,
				"sift_scores": sift_scores,
				"is_match": is_match,
				"matched_path": str(matched_path),
			}
		)

	return results


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_output_dir = args.output_dir / f"result_{run_ts}"
	run_output_dir.mkdir(parents=True, exist_ok=False)
	source_targets_dir = resolve_target_dir(args.targets_dir)
	targets_dir = ensure_target_images_dir(source_targets_dir, DEFAULT_TARGET_IMAGES_DIR, args.output_size)
	query_paths = resolve_query_images(args.query, args.warped_dir)

	matcher = TargetMatcher(targets_dir, device=args.device, output_size=args.output_size)
	sift_matcher = SIFTMatcher(targets_dir)
	results = match_query_images(
		matcher,
		sift_matcher,
		query_paths,
		run_output_dir,
		args.threshold,
		args.sift_top_k,
		args.sift_weight,
		args.warped_dir if args.query is None else args.query.parent,
	)

	matched_count = sum(1 for result in results if bool(result["is_match"]))
	top_result = max(results, key=lambda item: float(item["score"]))

	result = {
		"run_timestamp": run_ts,
		"source_targets_dir": str(source_targets_dir),
		"targets_dir": str(targets_dir),
		"query_root": str(args.warped_dir if args.query is None else args.query.parent),
		"query_count": len(results),
		"matched_count": matched_count,
		"match_rate": round(matched_count / len(results), 4),
		"threshold": args.threshold,
		"best_match": top_result,
		"results": results,
	}

	(run_output_dir / "result.json").write_text(
		json.dumps(result, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)

	print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
