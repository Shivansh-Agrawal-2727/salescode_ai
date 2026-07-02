from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features
from .io import iter_images, open_image

LABELS = ("real", "screen")


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    paths: list[Path]


def load_dataset(dataset_dir: Path) -> Dataset:
    xs: list[np.ndarray] = []
    ys: list[str] = []
    paths: list[Path] = []

    for label in LABELS:
        folder = dataset_dir / label
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for image_path in iter_images(folder):
            image = open_image(image_path)
            xs.append(extract_features(image))
            ys.append(label)
            paths.append(image_path)

    if not xs:
        raise ValueError(f"No images found in {dataset_dir}")

    return Dataset(np.vstack(xs), np.asarray(ys), paths)


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(dataset: Dataset, random_state: int = 42) -> tuple[Pipeline, dict]:
    model = build_model(random_state=random_state)
    stratify = dataset.labels if len(set(dataset.labels)) > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, labels=list(LABELS), zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(LABELS)).tolist(),
    }

    min_class_count = min(np.unique(dataset.labels, return_counts=True)[1])
    if min_class_count >= 3:
        folds = min(5, int(min_class_count))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, dataset.features, dataset.labels, cv=cv, scoring="accuracy")
        report["cross_validation_accuracy"] = {
            "folds": folds,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": [float(score) for score in scores],
        }

    model.fit(dataset.features, dataset.labels)
    return model, report


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "labels": LABELS}, output_path)


def load_model(model_path: Path) -> Pipeline:
    payload = joblib.load(model_path)
    return payload["model"]


def predict_image(model: Pipeline, image_path: Path) -> tuple[str, float, dict[str, float]]:
    image = open_image(image_path)
    features = extract_features(image).reshape(1, -1)
    label = str(model.predict(features)[0])
    probabilities = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, model.predict_proba(features)[0])
    }
    return label, probabilities[label], probabilities
