"""Airline Satisfaction AI — validated single-passenger prediction engine.

The public prediction API accepts one passenger as a dictionary or a
single-row pandas DataFrame. The saved model is loaded without retraining.
"""

from pathlib import Path
import json
import math

import joblib
import pandas as pd

from .config import (
    FEATURES as CONFIG_FEATURES,
    RATING_FEATURES,
    CATEGORICAL_FEATURES,
    CUSTOMER_TYPES,
    TRAVEL_TYPES,
    CLASS_TYPES,
    validate_rating,
    validate_age,
    validate_delay,
    validate_flight_distance,
)


# Paths and saved artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "airline_satisfaction_rf_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not METADATA_PATH.is_file():
    raise FileNotFoundError(f"Model metadata file not found: {METADATA_PATH}")

model = joblib.load(MODEL_PATH)
with METADATA_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

if "features" not in metadata:
    raise ValueError("model_metadata.json does not contain a 'features' field.")

METADATA_FEATURES = metadata["features"]
if METADATA_FEATURES != CONFIG_FEATURES:
    raise ValueError("Feature mismatch between config.py and model_metadata.json.")

FEATURES = METADATA_FEATURES
EXPECTED_CLASSES = {0, 1}

if not hasattr(model, "classes_"):
    raise ValueError("Loaded model does not expose 'classes_'.")

MODEL_CLASSES = list(model.classes_)
if set(MODEL_CLASSES) != EXPECTED_CLASSES:
    raise ValueError(
        f"Unexpected model classes: expected [0, 1], found {MODEL_CLASSES}."
    )


# Input conversion and validation
def prepare_passenger_data(passenger_data):
    """Return a new DataFrame from a dictionary or DataFrame input."""
    if isinstance(passenger_data, dict):
        frame = pd.DataFrame([passenger_data])
    elif isinstance(passenger_data, pd.DataFrame):
        frame = passenger_data.copy(deep=True)
    else:
        raise TypeError("passenger_data must be a dictionary or pandas DataFrame.")

    if frame.empty:
        raise ValueError("passenger_data cannot be empty.")
    return frame


def validate_features(passenger_data):
    """Check that every trained-model feature is supplied."""
    frame = prepare_passenger_data(passenger_data)
    missing_features = [name for name in FEATURES if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    return True


def _validate_numeric_values(passenger_data):
    """Reject missing, non-numeric, NaN, and infinite feature values."""
    for feature in FEATURES:
        values = pd.to_numeric(passenger_data[feature], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Feature '{feature}' contains missing or non-numeric values."
            )
        for value in values:
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Feature '{feature}' contains an infinite or non-finite value."
                )
    return True


def _validate_categorical_features(passenger_data):
    """Check integer category codes against configured mappings."""
    mappings = {
        "Customer Type": CUSTOMER_TYPES,
        "Type of Travel": TRAVEL_TYPES,
        "Class": CLASS_TYPES,
    }
    for feature in CATEGORICAL_FEATURES:
        if feature not in passenger_data.columns:
            continue

        valid_values = set(mappings[feature].values())
        values = pd.to_numeric(passenger_data[feature], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Categorical feature '{feature}' contains a missing "
                "or non-numeric value."
            )

        for value in values:
            if not math.isfinite(float(value)) or not float(value).is_integer():
                raise ValueError(
                    f"Categorical feature '{feature}' must contain "
                    "integer category codes."
                )
            integer_value = int(value)
            if integer_value not in valid_values:
                raise ValueError(
                    f"Invalid value '{integer_value}' for categorical feature "
                    f"'{feature}'. Expected one of {sorted(valid_values)}."
                )
    return True


def _validate_domain_values(passenger_data):
    """Apply the age, distance, delay, and rating limits in config.py."""
    if "Age" in passenger_data.columns:
        for value in passenger_data["Age"]:
            validate_age(value)

    if "Flight Distance" in passenger_data.columns:
        for value in passenger_data["Flight Distance"]:
            validate_flight_distance(value)

    for feature in RATING_FEATURES:
        if feature in passenger_data.columns:
            for value in passenger_data[feature]:
                validate_rating(value)

    for feature in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        if feature in passenger_data.columns:
            for value in passenger_data[feature]:
                validate_delay(value)
    return True


def prepare_model_input(passenger_data):
    """Validate one passenger and enforce exact model feature ordering."""
    frame = prepare_passenger_data(passenger_data)

    if len(frame) != 1:
        raise ValueError(
            "Prediction requires exactly one passenger. "
            "For multiple passengers, use a dedicated batch prediction function."
        )

    validate_features(frame)
    frame = frame[FEATURES].copy()
    _validate_numeric_values(frame)
    _validate_categorical_features(frame)
    _validate_domain_values(frame)
    return frame


# Class-aware prediction
def _get_class_probabilities(passenger_data):
    """Return (satisfied, dissatisfied) probabilities for one prepared row."""
    probabilities = model.predict_proba(passenger_data)[0]
    classes = model.classes_

    if len(probabilities) != len(classes):
        raise ValueError(
            "Model probability output does not match the number of model classes."
        )

    class_probability_map = {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }
    if 0 not in class_probability_map:
        raise ValueError("Model output does not contain class 0 (Neutral/Dissatisfied).")
    if 1 not in class_probability_map:
        raise ValueError("Model output does not contain class 1 (Satisfied).")

    dissatisfied_probability = class_probability_map[0]
    satisfied_probability = class_probability_map[1]

    for probability in class_probability_map.values():
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Model returned an invalid probability: {probability}")

    probability_sum = dissatisfied_probability + satisfied_probability
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Model probabilities do not sum to 1. Received sum: {probability_sum}"
        )
    return satisfied_probability, dissatisfied_probability


def predict_satisfaction(passenger_data):
    """Predict satisfaction for one passenger.

    `confidence` is the predicted class probability, not a calibrated
    estimate of how often the prediction will be correct.
    """
    prepared = prepare_model_input(passenger_data)
    prediction = int(model.predict(prepared)[0])
    if prediction not in EXPECTED_CLASSES:
        raise ValueError(f"Model returned an unexpected prediction class: {prediction}")

    satisfied_probability, dissatisfied_probability = _get_class_probabilities(
        prepared
    )
    label = "Satisfied" if prediction == 1 else "Neutral/Dissatisfied"
    confidence = (
        satisfied_probability if prediction == 1 else dissatisfied_probability
    )
    return {
        "prediction": prediction,
        "label": label,
        "satisfied_probability": satisfied_probability,
        "dissatisfied_probability": dissatisfied_probability,
        "confidence": confidence,
    }


# What-If comparison
def compare_predictions(baseline_data, what_if_data):
    """Compare predicted satisfaction probabilities for two scenarios.

    Differences are model simulations, not causal estimates of real-world
    improvements.
    """
    baseline = predict_satisfaction(baseline_data)
    what_if = predict_satisfaction(what_if_data)
    baseline_probability = baseline["satisfied_probability"]
    what_if_probability = what_if["satisfied_probability"]
    change = what_if_probability - baseline_probability
    return {
        "baseline": baseline,
        "what_if": what_if,
        "baseline_probability": baseline_probability,
        "what_if_probability": what_if_probability,
        "change": change,
        "change_percentage_points": change * 100,
    }


def get_probability_change(baseline_data, what_if_data):
    """Return the What-If satisfaction probability change in percentage points."""
    comparison = compare_predictions(baseline_data, what_if_data)
    return comparison["change_percentage_points"]
