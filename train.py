from __future__ import annotations

import argparse
import json
from pathlib import Path

from screen_real_detector.model import load_dataset, save_model, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a real photo vs screen photo classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Folder containing real/ and screen/ images.")
    parser.add_argument("--model", type=Path, default=Path("models/screen_real_classifier.joblib"), help="Model output path.")
    parser.add_argument("--report", type=Path, default=Path("reports/evaluation.json"), help="Evaluation report output path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    model, report = train_and_evaluate(dataset, random_state=args.seed)
    report["total_images"] = int(len(dataset.labels))
    report["class_counts"] = {label: int((dataset.labels == label).sum()) for label in sorted(set(dataset.labels))}

    save_model(model, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Trained on {report['total_images']} images")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Saved model: {args.model}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
