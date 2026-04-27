"""Run operational backend smoke scenarios without real model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

from app.service.operational_pipeline import OperationalInferenceService, OperationalPipelineConfig


SMOKE_SCENARIOS = [
	"normal_target2_accept",
	"unknown_no_pass",
	"ambiguous_reinspect",
	"poor_detector_reinspect",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run detector/preprocess/classifier/decision/state/db smoke orchestration")
	parser.add_argument("--scenario", choices=[*SMOKE_SCENARIOS, "all"], default="all")
	parser.add_argument("--mode", choices=["image", "sequence"], default="image")
	parser.add_argument("--model-mode", choices=["mock", "real"], default="mock")
	parser.add_argument("--width", type=int, default=640)
	parser.add_argument("--height", type=int, default=480)
	parser.add_argument("--frame-count", type=int, default=5)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	service = OperationalInferenceService(OperationalPipelineConfig(model_mode=args.model_mode))
	scenarios = SMOKE_SCENARIOS if args.scenario == "all" else [args.scenario]
	results = {
		scenario: service.run_smoke_scenario(
			scenario,
			width=args.width,
			height=args.height,
			mode=args.mode,
			frame_count=args.frame_count,
		)
		for scenario in scenarios
	}
	print(json.dumps({"mode": args.mode, "model_mode": args.model_mode, "db_path": str(service.config.db_path), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
