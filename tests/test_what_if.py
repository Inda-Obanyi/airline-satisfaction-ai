
import math

import pytest

from src.config import FEATURES
from src.predict import (
    predict_satisfaction,
    compare_predictions,
    get_probability_change,
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


def test_comparison_returns_expected_fields(passenger):
    result = compare_predictions(passenger, passenger.copy())

    assert set(result) == {
        "baseline",
        "what_if",
        "baseline_probability",
        "what_if_probability",
        "change",
        "change_percentage_points",
    }


def test_identical_scenarios_have_zero_change(passenger):
    result = compare_predictions(passenger, passenger.copy())

    assert result["change"] == pytest.approx(0.0)
    assert result["change_percentage_points"] == pytest.approx(0.0)


def test_comparison_probabilities_match_predictions(passenger):
    modified = passenger.copy()
    modified["Inflight entertainment"] = 5

    result = compare_predictions(passenger, modified)

    baseline = predict_satisfaction(passenger)
    simulated = predict_satisfaction(modified)

    assert result["baseline_probability"] == pytest.approx(
        baseline["satisfied_probability"]
    )
    assert result["what_if_probability"] == pytest.approx(
        simulated["satisfied_probability"]
    )


def test_probability_difference_arithmetic(passenger):
    modified = passenger.copy()
    modified["Seat comfort"] = 5

    result = compare_predictions(passenger, modified)

    expected_change = (
        result["what_if_probability"]
        - result["baseline_probability"]
    )

    assert result["change"] == pytest.approx(expected_change)
    assert result["change_percentage_points"] == pytest.approx(
        expected_change * 100
    )


def test_probability_change_helper(passenger):
    modified = passenger.copy()
    modified["Online support"] = 5

    comparison = compare_predictions(passenger, modified)

    helper_result = get_probability_change(passenger, modified)

    assert helper_result == pytest.approx(
        comparison["change_percentage_points"]
    )


def test_comparison_preserves_original_inputs(passenger):
    baseline = passenger.copy()
    modified = passenger.copy()
    modified["Inflight entertainment"] = 5

    original_baseline = baseline.copy()
    original_modified = modified.copy()

    compare_predictions(baseline, modified)

    assert baseline == original_baseline
    assert modified == original_modified


def test_comparison_rejects_invalid_scenario(passenger):
    modified = passenger.copy()
    modified["Seat comfort"] = 9

    with pytest.raises(ValueError):
        compare_predictions(passenger, modified)


def test_probability_change_is_bounded(passenger):
    modified = passenger.copy()
    modified["Inflight entertainment"] = 5

    change = get_probability_change(passenger, modified)

    assert math.isfinite(change)
    assert -100 <= change <= 100
