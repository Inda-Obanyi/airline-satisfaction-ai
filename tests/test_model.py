
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "airline_satisfaction_rf_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@pytest.fixture(scope="module")
def model_bundle():
    pipeline = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return pipeline, metadata


@pytest.fixture
def average_passenger(model_bundle):
    _, metadata = model_bundle

    passenger = {
        feature: 3 for feature in metadata["features"]
    }

    passenger.update({
        "Customer Type": 0,
        "Age": 35,
        "Type of Travel": 0,
        "Class": 1,
        "Flight Distance": 1000,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
    })

    return pd.DataFrame(
        [passenger],
        columns=metadata["features"],
    )


def test_model_is_trained(model_bundle):
    pipeline, _ = model_bundle

    classifier = pipeline.named_steps["classifier"]

    assert hasattr(classifier, "estimators_")
    assert len(classifier.estimators_) > 0


def test_tree_count_matches_metadata(model_bundle):
    pipeline, metadata = model_bundle

    classifier = pipeline.named_steps["classifier"]

    assert len(classifier.estimators_) == (
        metadata["model"]["n_estimators"]
    )


def test_feature_order_matches_metadata(model_bundle):
    pipeline, metadata = model_bundle

    assert list(pipeline.feature_names_in_) == (
        metadata["features"]
    )


def test_class_labels(model_bundle):
    pipeline, _ = model_bundle

    assert list(pipeline.classes_) == [0, 1]


def test_prediction_is_valid(model_bundle, average_passenger):
    pipeline, _ = model_bundle

    prediction = pipeline.predict(average_passenger)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]


def test_probabilities_sum_to_one(
    model_bundle,
    average_passenger,
):
    pipeline, _ = model_bundle

    probabilities = pipeline.predict_proba(
        average_passenger
    )

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(1),
        atol=1e-10,
    )


def test_prediction_matches_probability_argmax(
    model_bundle,
    average_passenger,
):
    pipeline, _ = model_bundle

    probabilities = pipeline.predict_proba(
        average_passenger
    )

    predicted_class = pipeline.predict(
        average_passenger
    )[0]

    expected_class = pipeline.classes_[
        np.argmax(probabilities[0])
    ]

    assert predicted_class == expected_class


def test_input_is_not_modified(
    model_bundle,
    average_passenger,
):
    pipeline, _ = model_bundle

    original = average_passenger.copy(deep=True)

    pipeline.predict_proba(average_passenger)

    pd.testing.assert_frame_equal(
        average_passenger,
        original,
    )


def test_repeat_prediction_is_deterministic(
    model_bundle,
    average_passenger,
):
    pipeline, _ = model_bundle

    first = pipeline.predict_proba(
        average_passenger
    )

    second = pipeline.predict_proba(
        average_passenger
    )

    np.testing.assert_allclose(
        first,
        second,
        atol=1e-12,
    )
