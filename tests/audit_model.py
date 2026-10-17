
from pathlib import Path
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURES as APP_FEATURES


MODEL_PATH = ROOT / "models" / "airline_satisfaction_rf_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


def main():
    print("\n" + "=" * 65)
    print("AIRLINE SATISFACTION AI — MODEL INTEGRITY AUDIT")
    print("=" * 65)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    print(f"\nModel file: {MODEL_PATH.name}")
    print(f"Model size: {MODEL_PATH.stat().st_size / (1024 ** 2):.2f} MB")
    print(f"Pipeline steps: {list(pipeline.named_steps)}")

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    actual_trees = classifier.n_estimators
    metadata_trees = metadata.get("model", {}).get("n_estimators")

    print(f"\nClassifier: {type(classifier).__name__}")
    print(f"Actual tree count: {actual_trees}")
    print(f"Metadata tree count: {metadata_trees}")
    print(f"Trained tree count: {len(classifier.estimators_)}")
    print(f"Class labels: {classifier.classes_.tolist()}")
    print(f"Random state: {classifier.random_state}")

    print(f"\nPreprocessor: {type(preprocessor).__name__}")
    print(f"Metadata features: {len(metadata.get('features', []))}")
    print(f"Application features: {len(APP_FEATURES)}")

    actual_features = list(getattr(pipeline, "feature_names_in_", []))
    metadata_features = metadata.get("features", [])

    checks = {
        "Actual tree count matches trained trees":
            actual_trees == len(classifier.estimators_),

        "Tree count matches metadata":
            actual_trees == metadata_trees,

        "Application feature order matches metadata":
            APP_FEATURES == metadata_features,

        "Model feature order matches metadata":
            actual_features == metadata_features,

        "Binary classes are [0, 1]":
            np.array_equal(classifier.classes_, [0, 1]),

        "Pipeline contains preprocessor and classifier":
            list(pipeline.named_steps) == [
                "preprocessor", "classifier"
            ],
    }

    print("\n" + "-" * 65)
    print("INTEGRITY CHECKS")
    print("-" * 65)

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    failed = [name for name, passed in checks.items() if not passed]

    print("\n" + "=" * 65)

    if failed:
        print(f"AUDIT FAILED — {len(failed)} issue(s) detected")
        sys.exit(1)

    print("AUDIT PASSED — model and metadata are consistent")


if __name__ == "__main__":
    main()
