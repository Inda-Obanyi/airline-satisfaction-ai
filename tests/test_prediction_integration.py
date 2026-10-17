
import math

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES
from src.predict import (
    model,
    prepare_model_input,
    predict_satisfaction,
    compare_predictions,
)


@pytest.fixture
def passenger():
    data = {feature: 3 for feature in FEATURES}

    data.update({
        "Customer Type": 0,
        "Age": 35,
        "Type of Travel": 0,
        "Class": 1,
        "Flight Distance": 1000,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
    })

    return data


def test_public_prediction_returns_required_fields(passenger):
    result = predict_satisfaction(passenger)

    required = {
        "prediction",
        "label",
        "satisfied_probability",
        "dissatisfied_probability",
        "confidence",
    }

    assert required.issubset(result)


def test_public_prediction_probability_consistency(passenger):
    result = predict_satisfaction(passenger)

    satisfied = result["satisfied_probability"]
    dissatisfied = result["dissatisfied_probability"]

    assert 0 <= satisfied <= 1
    assert 0 <= dissatisfied <= 1

    assert math.isclose(
        satisfied + dissatisfied,
        1.0,
        abs_tol=1e-10,
    )


def test_public_prediction_class_consistency(passenger):
    result = predict_satisfaction(passenger)

    expected_label = (
        "Satisfied"
        if result["prediction"] == 1
        else "Neutral/Dissatisfied"
    )

    assert result["label"] == expected_label

    expected_confidence = (
        result["satisfied_probability"]
        if result["prediction"] == 1
        else result["dissatisfied_probability"]
    )

    assert math.isclose(
        result["confidence"],
        expected_confidence,
        abs_tol=1e-10,
    )


def test_public_prediction_matches_saved_model(passenger):
    prepared = prepare_model_input(passenger)

    expected_prediction = int(model.predict(prepared)[0])

    probabilities = model.predict_proba(prepared)[0]

    probability_map = dict(
        zip(model.classes_, probabilities)
    )

    result = predict_satisfaction(passenger)

    assert result["prediction"] == expected_prediction

    assert math.isclose(
        result["satisfied_probability"],
        probability_map[1],
        abs_tol=1e-10,
    )


def test_feature_order_is_enforced(passenger):
    reversed_passenger = dict(
        reversed(list(passenger.items()))
    )

    prepared = prepare_model_input(reversed_passenger)

    assert list(prepared.columns) == FEATURES


def test_extra_columns_are_excluded(passenger):
    passenger["Unrelated Feature"] = 999

    prepared = prepare_model_input(passenger)

    assert "Unrelated Feature" not in prepared.columns
    assert list(prepared.columns) == FEATURES


@pytest.mark.parametrize(
    "feature,value",
    [
        ("Age", -1),
        ("Age", 121),
        ("Flight Distance", -100),
        ("Flight Distance", 10001),
        ("Departure Delay in Minutes", -1),
        ("Arrival Delay in Minutes", 2001),
        ("Seat comfort", -1),
        ("Seat comfort", 6),
        ("Inflight entertainment", 2.5),
        ("Customer Type", 9),
        ("Type of Travel", 9),
        ("Class", 9),
        ("Age", math.nan),
        ("Flight Distance", math.inf),
    ],
)
def test_public_prediction_rejects_invalid_inputs(
    passenger,
    feature,
    value,
):
    passenger[feature] = value

    with pytest.raises(ValueError):
        predict_satisfaction(passenger)


def test_public_prediction_rejects_missing_feature(passenger):
    passenger.pop("Inflight entertainment")

    with pytest.raises(ValueError):
        predict_satisfaction(passenger)


def test_public_prediction_preserves_input(passenger):
    original = passenger.copy()

    predict_satisfaction(passenger)

    assert passenger == original


def test_public_prediction_accepts_dataframe(passenger):
    frame = pd.DataFrame([passenger])

    result = predict_satisfaction(frame)

    assert result["prediction"] in [0, 1]


def test_public_prediction_is_repeatable(passenger):
    first = predict_satisfaction(passenger)
    second = predict_satisfaction(passenger)

    assert first["prediction"] == second["prediction"]

    np.testing.assert_allclose(
        first["satisfied_probability"],
        second["satisfied_probability"],
    )


def test_no_change_simulation(passenger):
    baseline = predict_satisfaction(passenger)

    simulated = predict_satisfaction(passenger.copy())

    assert math.isclose(
        baseline["satisfied_probability"],
        simulated["satisfied_probability"],
        abs_tol=1e-12,
    )


def test_single_factor_simulation_preserves_baseline(passenger):
    baseline_data = passenger.copy()
    simulated_data = passenger.copy()

    simulated_data["Inflight entertainment"] = 5

    original = baseline_data.copy()

    comparison = compare_predictions(
        baseline_data,
        simulated_data,
    )

    assert baseline_data == original
    assert isinstance(comparison, dict)


def test_identical_scenarios_compare_successfully(passenger):
    comparison = compare_predictions(
        passenger,
        passenger.copy(),
    )

    assert isinstance(comparison, dict)
