from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) in sys.path:
	sys.path.remove(str(SCRIPT_DIR))

try:
	import openvino as ov
except ImportError:  # pragma: no cover - optional runtime dependency
	ov = None


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ResNet18_Weights.DEFAULT
DEFAULT_INPUT_SIZE = 640
DEFAULT_THRESHOLD = 0.5
TARGET_THRESHOLDS: dict[str, float] = {
	"target_1": 0.90,
	"target_2": 0.75,
	"target_3": 0.90,
	"target_4": 0.90,
}

TARGET_SEQUENCE_SETTINGS: dict[str, dict[str, float | int]] = {
	"target_1": {"min_consecutive": 3, "sample_seconds": 0.5, "max_missed": 4},
	"target_2": {"min_consecutive": 3, "sample_seconds": 0.5, "max_missed": 4},
	"target_3": {"min_consecutive": 3, "sample_seconds": 0.5, "max_missed": 4},
	"target_4": {"min_consecutive": 2, "sample_seconds": 0.5, "max_missed": 4},
}


@dataclass(slots=True)
class BinaryPrediction:
	label: str
	score: float
	prob_yes: float
	prob_no: float


def load_bgr_image(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path))
	if image is None:
		raise RuntimeError(f"failed to read image: {image_path}")
	return image


def preprocess_for_resnet(image_bgr: np.ndarray, output_size: int = DEFAULT_INPUT_SIZE) -> torch.Tensor:
	if image_bgr.size == 0:
		raise ValueError("empty image input")
	height, width = image_bgr.shape[:2]
	if height != output_size or width != output_size:
		raise ValueError(f"expected {output_size}x{output_size} warped input, got {width}x{height}")
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
	mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
	return (tensor - mean) / std


def preprocess_for_openvino(image_bgr: np.ndarray, output_size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
	return preprocess_for_resnet(image_bgr, output_size=output_size).unsqueeze(0).cpu().numpy().astype(np.float32)


class TargetResNet18(nn.Module):
	def __init__(self, *, pretrained: bool = True, freeze_backbone: bool = False, num_classes: int = 2) -> None:
		super().__init__()
		weights = DEFAULT_WEIGHTS if pretrained else None
		backbone = resnet18(weights=weights)
		in_features = backbone.fc.in_features
		backbone.fc = nn.Linear(in_features, num_classes)
		self.backbone = backbone
		self.num_classes = num_classes

		if freeze_backbone:
			for name, parameter in self.backbone.named_parameters():
				if not name.startswith("fc."):
					parameter.requires_grad = False

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.backbone(inputs)

	@torch.no_grad()
	def predict(self, inputs: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> BinaryPrediction:
		self.eval()
		logits = self.forward(inputs)
		if logits.ndim == 1:
			logits = logits.unsqueeze(0)
		probabilities = torch.softmax(logits, dim=1)
		prob_no = float(probabilities[0, 0].item())
		prob_yes = float(probabilities[0, 1].item())
		label = "yes" if prob_yes >= threshold else "no"
		score = prob_yes if label == "yes" else prob_no
		return BinaryPrediction(label=label, score=score, prob_yes=prob_yes, prob_no=prob_no)

	@torch.no_grad()
	def predict_bgr(self, image_bgr: np.ndarray, *, device: str = "cpu", threshold: float = DEFAULT_THRESHOLD) -> BinaryPrediction:
		input_tensor = preprocess_for_resnet(image_bgr).unsqueeze(0).to(device)
		self = self.to(device)
		return self.predict(input_tensor, threshold=threshold)


class OpenVINOTargetResNet18:
	def __init__(self, model_path: Path, *, device: str = "CPU") -> None:
		if ov is None:
			raise ImportError("openvino is not installed")
		self.model_path = Path(model_path)
		self.device = device
		self.core = ov.Core()
		self.model = self.core.read_model(str(self.model_path))
		self.compiled_model = self.core.compile_model(self.model, device)
		self.input_port = self.compiled_model.inputs[0]
		self.output_port = self.compiled_model.outputs[0]

	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		logits = logits.astype(np.float32)
		logits = logits - np.max(logits, axis=-1, keepdims=True)
		exp_logits = np.exp(logits)
		return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

	@torch.no_grad()
	def predict(self, inputs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> BinaryPrediction:
		if inputs.ndim == 3:
			inputs = np.expand_dims(inputs, axis=0)
		if inputs.dtype != np.float32:
			inputs = inputs.astype(np.float32)
		infer_request = self.compiled_model.create_infer_request()
		infer_request.infer({self.input_port: inputs})
		logits = np.array(infer_request.get_output_tensor(0).data, dtype=np.float32)
		if logits.ndim == 1:
			logits = np.expand_dims(logits, axis=0)
		probabilities = self._softmax(logits)
		prob_no = float(probabilities[0, 0].item())
		prob_yes = float(probabilities[0, 1].item())
		label = "yes" if prob_yes >= threshold else "no"
		score = prob_yes if label == "yes" else prob_no
		return BinaryPrediction(label=label, score=score, prob_yes=prob_yes, prob_no=prob_no)

	@torch.no_grad()
	def predict_bgr(
		self,
		image_bgr: np.ndarray,
		*,
		device: str | None = None,
		threshold: float = DEFAULT_THRESHOLD,
	) -> BinaryPrediction:
		input_array = preprocess_for_openvino(image_bgr)
		return self.predict(input_array, threshold=threshold)


def build_target_model(*, pretrained: bool = True, freeze_backbone: bool = False) -> TargetResNet18:
	return TargetResNet18(pretrained=pretrained, freeze_backbone=freeze_backbone, num_classes=2)


def load_target_model(weights_path: Path | None = None, *, device: str = "cpu") -> TargetResNet18:
	model = build_target_model(pretrained=False)
	if weights_path is not None:
		state_dict = torch.load(weights_path, map_location=device)
		model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model


def resolve_openvino_model_path(model_path: Path) -> Path:
	if model_path.is_dir():
		candidate = model_path / "model.xml"
		if candidate.exists():
			return candidate
		raise FileNotFoundError(f"OpenVINO model.xml not found in directory: {model_path}")
	if model_path.suffix.lower() == ".xml":
		return model_path
	raise ValueError(f"Unsupported OpenVINO model path: {model_path}")


def load_openvino_target_model(model_path: Path, *, device: str = "CPU") -> OpenVINOTargetResNet18:
	resolved_model_path = resolve_openvino_model_path(model_path)
	return OpenVINOTargetResNet18(resolved_model_path, device=device)


def get_target_threshold(target_name: str, default: float = DEFAULT_THRESHOLD) -> float:
	return TARGET_THRESHOLDS.get(target_name, default)


def get_target_sequence_settings(target_name: str) -> dict[str, float | int]:
	return TARGET_SEQUENCE_SETTINGS.get(target_name, {})


Target1ResNet18 = TargetResNet18
build_target_1_model = build_target_model
load_target_1_model = load_target_model
