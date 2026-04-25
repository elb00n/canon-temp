"""Convert trained target weight files into OpenVINO IR models.

The script scans `outputs/cnn_train/*_weight/best.pt` and writes each
converted model into a sibling directory named `*_weight_openvino`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import resnet18

from app.core.paths import ASSET_WEIGHTS_DIR, asset_openvino_dir

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
	sys.path.remove(str(SCRIPT_DIR))

try:
	import openvino as ov
except ImportError as exc:  # pragma: no cover - handled at runtime
	raise ImportError(
		"openvino is not installed. Install it with `uv pip install openvino` or the project environment manager."
	) from exc
DEFAULT_TARGET_ROOT = ASSET_WEIGHTS_DIR
DEFAULT_EXAMPLE_SIZE = 640


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert target ResNet18 weights to OpenVINO IR")
	parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
	parser.add_argument("--input-size", type=int, default=DEFAULT_EXAMPLE_SIZE)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing OpenVINO output folders")
	return parser.parse_args()


def list_weight_files(target_root: Path) -> list[Path]:
	if not target_root.exists():
		return []
	weight_files: list[Path] = []
	for target_dir in sorted(path for path in target_root.iterdir() if path.is_dir() and path.name.startswith("target_")):
		best_path = target_dir / "best.pt"
		if not best_path.exists():
			best_path = target_dir / "weights" / "best.pt"
		if best_path.exists():
			weight_files.append(best_path)
	return weight_files


def build_model(num_classes: int = 2) -> torch.nn.Module:
	model = resnet18(weights=None)
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
	return model


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	normalized_state_dict: dict[str, torch.Tensor] = {}
	for key, value in state_dict.items():
		if key.startswith("backbone."):
			normalized_state_dict[key.removeprefix("backbone.")] = value
		else:
			normalized_state_dict[key] = value
	return normalized_state_dict


def resolve_output_dir(weight_path: Path) -> Path:
	weight_dir = weight_path.parent
	return asset_openvino_dir(weight_dir.name)


def convert_weight(weight_path: Path, input_size: int, overwrite: bool) -> Path:
	output_dir = resolve_output_dir(weight_path)
	model_xml = output_dir / "model.xml"
	model_bin = output_dir / "model.bin"

	if model_xml.exists() and model_bin.exists() and not overwrite:
		return output_dir

	loaded_object = torch.load(weight_path, map_location="cpu")
	state_dict = loaded_object.get("state_dict", loaded_object) if isinstance(loaded_object, dict) else loaded_object
	if not isinstance(state_dict, dict):
		raise TypeError(f"Unsupported checkpoint format in {weight_path}: expected a state_dict-like mapping")
	state_dict = normalize_state_dict_keys(state_dict)
	model = build_model()
	model.load_state_dict(state_dict)
	model.eval()

	example_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
	openvino_model = ov.convert_model(model, example_input=example_input)
	output_dir.mkdir(parents=True, exist_ok=True)
	ov.save_model(openvino_model, output_dir / "model.xml", compress_to_fp16=False)
	return output_dir


def main() -> None:
	args = parse_args()
	weight_files = list_weight_files(args.target_root)
	if not weight_files:
		raise FileNotFoundError(f"No weight files found under {args.target_root}")

	for weight_path in weight_files:
		output_dir = convert_weight(weight_path, args.input_size, args.overwrite)
		print(f"Converted {weight_path} -> {output_dir}")


if __name__ == "__main__":
	main()
