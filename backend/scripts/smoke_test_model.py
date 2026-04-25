"""Quick model smoke test without training."""

from __future__ import annotations

import numpy as np

from app.models import FramePredictor


def main() -> None:
    predictor = FramePredictor(backbone="mobilenet_v3_small")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = predictor.predict_bgr(dummy_frame)

    print("Predicted:", result.label)
    print("Confidence:", round(result.confidence, 4))
    print("Probabilities:", result.probabilities)


if __name__ == "__main__":
    main()
