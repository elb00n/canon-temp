"""Train a ResNet18 binary classifier for any target using yes/no image folders."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.core.paths import ASSET_WEIGHTS_DIR, CNN_TRAIN_DIR
from app.models.target_model import DEFAULT_INPUT_SIZE, TargetResNet18, load_bgr_image, preprocess_for_resnet


DEFAULT_DATA_ROOT = CNN_TRAIN_DIR
DEFAULT_OUTPUT_ROOT = ASSET_WEIGHTS_DIR
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_EARLY_STOP_PATIENCE = 4
DEFAULT_EARLY_STOP_MIN_DELTA = 1e-4
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class SampleItem:
	image_path: str
	label: int


@dataclass(slots=True)
class TrainSummary:
	run_timestamp: str
	target_name: str
	data_root: str
	yes_dir: str
	no_dir: str
	output_dir: str
	best_weights: str
	train_size: int
	val_size: int
	epochs: int
	batch_size: int
	lr: float
	weight_decay: float
	val_ratio: float
	early_stop_patience: int
	early_stop_min_delta: float
	stopped_epoch: int
	best_val_loss: float
	best_val_accuracy: float
	device: str
	input_size: int


class TargetBinaryDataset(Dataset):
	def __init__(self, samples: list[SampleItem], input_size: int = DEFAULT_INPUT_SIZE) -> None:
		self.samples = samples
		self.input_size = input_size

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		sample = self.samples[index]
		image_bgr = load_bgr_image(Path(sample.image_path))
		height, width = image_bgr.shape[:2]
		if height != self.input_size or width != self.input_size:
			image_bgr = torch.from_numpy(image_bgr).numpy()  # keep a safe copy path before resize
			import cv2
			image_bgr = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
		image_tensor = preprocess_for_resnet(image_bgr, output_size=self.input_size)
		label_tensor = torch.tensor(sample.label, dtype=torch.long)
		return image_tensor, label_tensor


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train ResNet18 target yes/no classifier")
	parser.add_argument("--target-name", type=str, required=True, help="Target folder name, e.g. target_1")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--lr", type=float, default=DEFAULT_LR)
	parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
	parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
	parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
	parser.add_argument("--freeze-backbone", action="store_true")
	return parser.parse_args()


def list_image_files(root: Path) -> list[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def build_samples(yes_dir: Path, no_dir: Path) -> list[SampleItem]:
	yes_paths = list_image_files(yes_dir)
	no_paths = list_image_files(no_dir)
	if not yes_paths:
		raise FileNotFoundError(f"no yes images found in {yes_dir}")
	if not no_paths:
		raise FileNotFoundError(f"no no images found in {no_dir}")
	samples = [SampleItem(image_path=str(path), label=1) for path in yes_paths]
	samples.extend(SampleItem(image_path=str(path), label=0) for path in no_paths)
	return samples


def split_samples(samples: list[SampleItem], val_ratio: float, seed: int) -> tuple[list[SampleItem], list[SampleItem]]:
	if not 0.0 < val_ratio < 1.0:
		raise ValueError("val-ratio must be between 0 and 1")
	indices = list(range(len(samples)))
	random.Random(seed).shuffle(indices)
	val_size = max(1, int(round(len(samples) * val_ratio)))
	val_indices = set(indices[:val_size])
	train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
	val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
	if not train_samples or not val_samples:
		raise RuntimeError("train/val split produced an empty partition")
	return train_samples, val_samples


def make_loader(samples: list[SampleItem], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
	dataset = TargetBinaryDataset(samples)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
	predictions = torch.argmax(logits, dim=1)
	return float((predictions == targets).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
	model.eval()
	total_loss = 0.0
	total_accuracy = 0.0
	count = 0
	for inputs, targets in loader:
		inputs = inputs.to(device)
		targets = targets.to(device)
		logits = model(inputs)
		loss = criterion(logits, targets)
		batch_size = inputs.size(0)
		total_loss += float(loss.item()) * batch_size
		total_accuracy += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
	if count == 0:
		return 0.0, 0.0
	return total_loss / count, total_accuracy / count


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
	model.train()
	total_loss = 0.0
	total_accuracy = 0.0
	count = 0
	for inputs, targets in loader:
		inputs = inputs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(inputs)
		loss = criterion(logits, targets)
		loss.backward()
		optimizer.step()
		batch_size = inputs.size(0)
		total_loss += float(loss.item()) * batch_size
		total_accuracy += accuracy_from_logits(logits, targets) * batch_size
		count += batch_size
	if count == 0:
		return 0.0, 0.0
	return total_loss / count, total_accuracy / count


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
	weights_dir = output_dir / "weights"
	log_dir = output_dir / "logs"
	for path in (output_dir, weights_dir, log_dir):
		path.mkdir(parents=True, exist_ok=True)
	return {"root": output_dir, "weights": weights_dir, "logs": log_dir}


def run_training(args: argparse.Namespace) -> TrainSummary:
	yes_dir = args.data_root / args.target_name
	no_dir = args.data_root / f"{args.target_name}_no"
	if not yes_dir.exists():
		raise FileNotFoundError(f"yes directory not found: {yes_dir}")
	if not no_dir.exists():
		raise FileNotFoundError(f"no directory not found: {no_dir}")

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	samples = build_samples(yes_dir, no_dir)
	train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
	train_loader = make_loader(train_samples, args.batch_size, shuffle=True, num_workers=args.num_workers)
	val_loader = make_loader(val_samples, args.batch_size, shuffle=False, num_workers=args.num_workers)

	device = torch.device(args.device)
	model = TargetResNet18(pretrained=True, freeze_backbone=args.freeze_backbone, num_classes=2).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

	run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	output_dir = args.output_root / args.target_name / run_ts
	output_dirs = ensure_output_dirs(output_dir)
	best_weights_path = output_dirs["weights"] / "best.pt"
	stable_weights_dir = args.output_root / args.target_name
	stable_weights_dir.mkdir(parents=True, exist_ok=True)
	stable_weights_path = stable_weights_dir / "best.pt"
	best_val_loss = float("inf")
	best_val_accuracy = 0.0
	best_epoch = 0
	patience_counter = 0
	train_history: list[dict[str, float]] = []

	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		scheduler.step(val_loss)
		train_history.append(
			{
				"epoch": float(epoch),
				"train_loss": train_loss,
				"train_accuracy": train_acc,
				"val_loss": val_loss,
				"val_accuracy": val_acc,
			}
		)

		if val_loss < (best_val_loss - args.early_stop_min_delta):
			best_val_loss = val_loss
			best_val_accuracy = val_acc
			best_epoch = epoch
			patience_counter = 0
			torch.save(model.state_dict(), best_weights_path)
			shutil.copy2(best_weights_path, stable_weights_path)
		else:
			patience_counter += 1

		print(
			f"epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
		)

		if patience_counter >= args.early_stop_patience:
			print(
				f"early stopping at epoch {epoch:03d} | best_epoch={best_epoch:03d} | best_val_loss={best_val_loss:.6f} | best_val_accuracy={best_val_accuracy:.4f}"
			)
			break

	summary = TrainSummary(
		run_timestamp=run_ts,
		target_name=args.target_name,
		data_root=str(args.data_root),
		yes_dir=str(yes_dir),
		no_dir=str(no_dir),
		output_dir=str(output_dir),
		best_weights=str(best_weights_path),
		train_size=len(train_samples),
		val_size=len(val_samples),
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		weight_decay=args.weight_decay,
		val_ratio=args.val_ratio,
		early_stop_patience=args.early_stop_patience,
		early_stop_min_delta=args.early_stop_min_delta,
		stopped_epoch=best_epoch,
		best_val_loss=best_val_loss,
		best_val_accuracy=best_val_accuracy,
		device=args.device,
		input_size=DEFAULT_INPUT_SIZE,
	)
	(output_dirs["root"] / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
	(output_dirs["logs"] / "history.json").write_text(json.dumps(train_history, indent=2, ensure_ascii=False), encoding="utf-8")
	return summary


def main() -> None:
	args = parse_args()
	summary = run_training(args)
	print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
