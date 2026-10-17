# =========================================================
# Airline Satisfaction AI
# Feature Sensitivity Diagnostic
# =========================================================

from copy import deepcopy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.config import RATING_FEATURES
from src.predict import predict_satisfaction


# ---------------------------------------------------------
# FIXED BASELINE PASSENGER
# ---------------------------------------------------------

baseline = {
    "Customer Type": 0,
    "Age": 35,
    "Type of Travel": 0,
    "Class": 0,
    "Flight Distance": 1000,

    "Seat comfort": 0,
    "Departure/Arrival time convenient": 0,
    "Food and drink": 0,
    "Gate location": 0,
    "Inflight wifi service": 0,
    "Inflight entertainment": 0,
    "Online support": 0,
    "Ease of Online booking": 0,
    "On-board service": 0,
    "Leg room service": 0,
    "Baggage handling": 0,
    "Checkin service": 0,
    "Cleanliness": 0,
    "Online boarding": 0,

    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0,
}


# ---------------------------------------------------------
# BASELINE PREDICTION
# ---------------------------------------------------------

baseline_result = predict_satisfaction(baseline)

print("=" * 70)
print("BASELINE PASSENGER")
print("=" * 70)

print(
    f"Satisfied probability: "
    f"{baseline_result['satisfied_probability']:.2%}"
)

print(
    f"Prediction: "
    f"{baseline_result['label']}"
)

print()


# ---------------------------------------------------------
# ONE-FEATURE-AT-A-TIME TEST
# ---------------------------------------------------------

print("=" * 70)
print("RATING FEATURE SENSITIVITY: 0 → 5")
print("=" * 70)

results = []

for feature in RATING_FEATURES:

    test_data = deepcopy(baseline)

    # Change ONLY this feature
    test_data[feature] = 5

    result = predict_satisfaction(test_data)

    baseline_probability = (
        baseline_result["satisfied_probability"]
    )

    new_probability = (
        result["satisfied_probability"]
    )

    change = (
        new_probability - baseline_probability
    ) * 100

    results.append(
        (
            feature,
            baseline_probability * 100,
            new_probability * 100,
            change,
        )
    )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

for feature, before, after, change in results:

    direction = (
        "↑"
        if change > 0
        else "↓"
        if change < 0
        else "→"
    )

    print(
        f"{feature:<42} "
        f"{before:6.2f}% → {after:6.2f}% "
        f"{direction} {change:+6.2f} pp"
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

positive = [
    item for item in results
    if item[3] > 0
]

negative = [
    item for item in results
    if item[3] < 0
]

unchanged = [
    item for item in results
    if item[3] == 0
]

print(f"Positive responses : {len(positive)}")
print(f"Negative responses : {len(negative)}")
print(f"Unchanged          : {len(unchanged)}")

print()
print("Largest positive responses:")

for item in sorted(
    results,
    key=lambda x: x[3],
    reverse=True
)[:5]:

    print(
        f"- {item[0]}: "
        f"{item[3]:+.2f} percentage points"
    )

print()
print("Largest negative responses:")

for item in sorted(
    results,
    key=lambda x: x[3]
)[:5]:

    print(
        f"- {item[0]}: "
        f"{item[3]:+.2f} percentage points"
    )