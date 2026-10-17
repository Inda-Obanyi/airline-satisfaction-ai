
import math

import pandas as pd
import pytest

from src.config import (
    FEATURES,
    RATING_FEATURES,
    AGE_MIN,
    AGE_MAX,
    FLIGHT_DISTANCE_MIN,
    FLIGHT_DISTANCE_MAX,
    DELAY_MIN,
    DELAY_MAX,
    validate_rating,
    validate_age,
    validate_flight_distance,
    validate_delay,
)

from src.predict import (
    prepare_passenger_data,
    validate_features,
    _validate_categorical_features,
    _validate_domain_values,
    _validate_numeric_values,
)


@pytest.fixture
def valid_passenger():
    passenger = {feature: 3 for feature in FEATURES}

    passenger.update({
        "Customer Type": 0,
        "Age": 35,
        "Type of Travel": 0,
        "Class": 1,
        "Flight Distance": 1000,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
    })

    return passenger


def test_valid_passenger(valid_passenger):
    frame = prepare_passenger_data(valid_passenger)

    assert validate_features(frame)
    assert _validate_categorical_features(frame)
    assert _validate_domain_values(frame)
    assert _validate_numeric_values(frame)


@pytest.mark.parametrize("rating", [0, 1, 2, 3, 4, 5])
def test_valid_ratings(rating):
    assert validate_rating(rating) == rating


@pytest.mark.parametrize(
    "rating",
    [-1, 6, 2.5, None, "invalid", math.nan, math.inf],
)
def test_invalid_ratings(rating):
    with pytest.raises(ValueError):
        validate_rating(rating)


@pytest.mark.parametrize("age", [AGE_MIN, 35, AGE_MAX])
def test_valid_ages(age):
    assert validate_age(age) is not None


@pytest.mark.parametrize("age", [0, 121, -5, None, "abc"])
def test_invalid_ages(age):
    with pytest.raises(ValueError):
        validate_age(age)


@pytest.mark.parametrize(
    "distance",
    [FLIGHT_DISTANCE_MIN, 1000, FLIGHT_DISTANCE_MAX],
)
def test_valid_flight_distances(distance):
    assert validate_flight_distance(distance) is not None


@pytest.mark.parametrize(
    "distance",
    [-1, 10001, None, "invalid"],
)
def test_invalid_flight_distances(distance):
    with pytest.raises(ValueError):
        validate_flight_distance(distance)


@pytest.mark.parametrize("delay", [DELAY_MIN, 30, DELAY_MAX])
def test_valid_delays(delay):
    assert validate_delay(delay) is not None


@pytest.mark.parametrize("delay", [-1, 2001, None, "invalid"])
def test_invalid_delays(delay):
    with pytest.raises(ValueError):
        validate_delay(delay)


def test_missing_required_feature(valid_passenger):
    passenger = valid_passenger.copy()
    passenger.pop("Age")

    with pytest.raises(ValueError, match="Missing required features"):
        validate_features(passenger)


@pytest.mark.parametrize(
    "feature,invalid_value",
    [
        ("Customer Type", 3),
        ("Type of Travel", 5),
        ("Class", 7),
        ("Class", 1.5),
        ("Customer Type", "unknown"),
    ],
)
def test_invalid_categories(
    valid_passenger,
    feature,
    invalid_value,
):
    passenger = valid_passenger.copy()
    passenger[feature] = invalid_value

    frame = prepare_passenger_data(passenger)

    with pytest.raises(ValueError):
        _validate_categorical_features(frame)


@pytest.mark.parametrize(
    "feature",
    [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ],
)
def test_non_finite_numeric_values(
    valid_passenger,
    feature,
):
    passenger = valid_passenger.copy()
    passenger[feature] = math.inf

    frame = prepare_passenger_data(passenger)

    with pytest.raises(ValueError):
        _validate_numeric_values(frame)


def test_empty_dataframe_rejected():
    with pytest.raises(ValueError):
        prepare_passenger_data(pd.DataFrame())


def test_unsupported_input_type_rejected():
    with pytest.raises(TypeError):
        prepare_passenger_data([1, 2, 3])


def test_dataframe_is_copied(valid_passenger):
    original = pd.DataFrame([valid_passenger])
    prepared = prepare_passenger_data(original)

    prepared.loc[0, "Age"] = 99

    assert original.loc[0, "Age"] == 35


@pytest.mark.parametrize("feature", RATING_FEATURES)
def test_every_rating_feature_is_validated(
    valid_passenger,
    feature,
):
    passenger = valid_passenger.copy()
    passenger[feature] = 6

    frame = prepare_passenger_data(passenger)

    with pytest.raises(ValueError):
        _validate_domain_values(frame)
