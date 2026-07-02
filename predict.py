from __future__ import annotations

import argparse
from pathlib import Path

from screen_real_detector.io import iter_images
from screen_real_detector.model import load_model, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether images are real photos or screen photos.")
    parser.add_argument("input", type=Path, help="Image file or folder to classify.")
    parser.add_argument("--model", type=Path, default=Path("models/screen_real_classifier.joblib"), help="Trained model path.")
    args = parser.parse_args()

    model = load_model(args.model)
    paths = list(iter_images(args.input)) if args.input.is_dir() else [args.input]

    for image_path in paths:
        label, confidence, probabilities = predict_image(model, image_path)
        real = probabilities.get("real", 0.0)
        screen = probabilities.get("screen", 0.0)
        print(f"{image_path}\t{label}\tconfidence={confidence:.3f}\treal={real:.3f}\tscreen={screen:.3f}")


if __name__ == "__main__":
    main()
