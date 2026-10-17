# =========================================================
# Airline Satisfaction AI
# Application Configuration
# =========================================================


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = [
    "Customer Type",
    "Age",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


# =========================================================
# FEATURE GROUPS
# =========================================================

CATEGORICAL_FEATURES = [
    "Customer Type",
    "Type of Travel",
    "Class",
]


PASSENGER_PROFILE_FEATURES = [
    "Customer Type",
    "Age",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


RATING_FEATURES = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]


ACTIONABLE_FEATURES = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


# =========================================================
# CATEGORY MAPPINGS
# =========================================================

CUSTOMER_TYPES = {
    "Loyal Customer": 0,
    "Disloyal Customer": 1,
}


TRAVEL_TYPES = {
    "Business travel": 0,
    "Personal Travel": 1,
}


CLASS_TYPES = {
    "Business": 0,
    "Eco": 1,
    "Eco Plus": 2,
}


# =========================================================
# TARGET
# =========================================================

TARGET_LABELS = {
    0: "Neutral/Dissatisfied",
    1: "Satisfied",
}


# =========================================================
# VALIDATION SETTINGS
# =========================================================

# ---------------------------------------------------------
# Experience ratings
# ---------------------------------------------------------

# Full raw dataset rating range.
#
# 0 = N/A / not-rated
# 1 = lowest scored rating
# 5 = highest scored rating

RATING_MIN = 0
RATING_MAX = 5

SPECIAL_RATING_VALUE = 0

# Actual scored rating range.
RATING_SCALE_MIN = 1
RATING_SCALE_MAX = 5


# ---------------------------------------------------------
# Passenger age
# ---------------------------------------------------------

AGE_MIN = 1
AGE_MAX = 120


# ---------------------------------------------------------
# Flight distance
# ---------------------------------------------------------

# Application-level safety boundary.
#
# This prevents clearly unrealistic values from being
# submitted to the trained model while allowing normal
# commercial flight distances.

FLIGHT_DISTANCE_MIN = 0
FLIGHT_DISTANCE_MAX = 10000


# ---------------------------------------------------------
# Flight delays
# ---------------------------------------------------------

# Delays cannot be negative.

DELAY_MIN = 0

# Application-level safety boundary.
#
# This protects the model from extreme out-of-distribution
# inputs such as millions of minutes.
#
# This is a validation boundary, NOT a model threshold.

DELAY_MAX = 2000


# =========================================================
# BUSINESS PRIORITY SETTINGS
# =========================================================

# Maximum delay considered when normalizing the business
# improvement opportunity.
#
# This is NOT:
#   - a model training parameter
#   - a prediction threshold
#   - a causal estimate
#
# It is only used by the business-priority heuristic.
#
# Example:
#
#   0 minutes    -> 0.00 opportunity
#   30 minutes   -> 0.17 opportunity
#   60 minutes   -> 0.33 opportunity
#   120 minutes  -> 0.67 opportunity
#   180 minutes  -> 1.00 opportunity
#   300 minutes  -> 1.00 opportunity
#
# Delays above the cap are intentionally limited to 1.0
# so that very large delays do not dominate the priority
# score simply because they are measured in minutes.

DELAY_OPPORTUNITY_CAP_MINUTES = 180.0


# =========================================================
# VALIDATION FUNCTIONS
# =========================================================

def validate_rating(value):
    """
    Validate a passenger experience rating.

    Valid values:

        0 -> N/A / not-rated
        1 -> lowest scored rating
        2
        3
        4
        5 -> highest scored rating

    Ratings are required to be whole numbers because the
    original airline survey uses discrete rating values.
    """

    if value is None:
        raise ValueError(
            "Rating cannot be empty."
        )

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Rating must be a valid number."
        )

    if not RATING_MIN <= numeric_value <= RATING_MAX:
        raise ValueError(
            f"Rating must be between "
            f"{RATING_MIN} and "
            f"{RATING_MAX}."
        )

    if not numeric_value.is_integer():
        raise ValueError(
            "Rating must be a whole number from "
            f"{RATING_MIN} to {RATING_MAX}."
        )

    return int(numeric_value)


def format_rating(value):
    """
    Format airline survey ratings.

    In the Invistico Airline dataset, 0 behaves as a
    special N/A / not-rated value rather than a 0/5 rating.
    """

    value = float(value)

    if value == SPECIAL_RATING_VALUE:
        return "Not rated / N/A"

    return f"{int(value)}/5"


def validate_age(value):
    """
    Validate passenger age.
    """

    if value is None:
        raise ValueError(
            "Age cannot be empty."
        )

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Age must be a valid whole number."
        )

    if not numeric_value.is_integer():
        raise ValueError(
            "Age must be a whole number."
        )

    value = int(numeric_value)

    if not AGE_MIN <= value <= AGE_MAX:
        raise ValueError(
            f"Age must be between "
            f"{AGE_MIN} and "
            f"{AGE_MAX}."
        )

    return value


def validate_flight_distance(value):
    """
    Validate flight distance in kilometers.

    The validation protects the model from clearly
    unrealistic out-of-distribution values.
    """

    if value is None:
        raise ValueError(
            "Flight distance cannot be empty."
        )

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Flight distance must be a valid number."
        )

    if not numeric_value == numeric_value:
        raise ValueError(
            "Flight distance cannot be NaN."
        )

    if numeric_value == float("inf") or numeric_value == float("-inf"):
        raise ValueError(
            "Flight distance must be finite."
        )

    if not (
        FLIGHT_DISTANCE_MIN
        <= numeric_value
        <= FLIGHT_DISTANCE_MAX
    ):
        raise ValueError(
            f"Flight distance must be between "
            f"{FLIGHT_DISTANCE_MIN} and "
            f"{FLIGHT_DISTANCE_MAX} km."
        )

    return numeric_value


def validate_delay(value):
    """
    Validate flight delay in minutes.

    Valid range:

        0 -> no delay
        DELAY_MAX -> maximum accepted application input

    This is an application safety boundary and not a model
    prediction threshold.
    """

    if value is None:
        raise ValueError(
            "Delay cannot be empty."
        )

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        raise ValueError(
            "Delay must be a valid number."
        )

    if numeric_value != numeric_value:
        raise ValueError(
            "Delay cannot be NaN."
        )

    if numeric_value == float("inf") or numeric_value == float("-inf"):
        raise ValueError(
            "Delay must be finite."
        )

    if numeric_value < DELAY_MIN:
        raise ValueError(
            "Delay cannot be negative."
        )

    if numeric_value > DELAY_MAX:
        raise ValueError(
            f"Delay cannot exceed "
            f"{DELAY_MAX} minutes."
        )

    return numeric_value


# =========================================================
# DISPLAY NAMES
# =========================================================

DISPLAY_NAMES = {
    "Customer Type":
        "Customer type",

    "Age":
        "Age",

    "Type of Travel":
        "Travel type",

    "Class":
        "Travel class",

    "Flight Distance":
        "Flight distance",

    "Seat comfort":
        "Seat comfort",

    "Departure/Arrival time convenient":
        "Schedule convenience",

    "Food and drink":
        "Food & drink",

    "Gate location":
        "Gate location",

    "Inflight wifi service":
        "Inflight Wi-Fi",

    "Inflight entertainment":
        "Inflight entertainment",

    "Online support":
        "Online support",

    "Ease of Online booking":
        "Online booking",

    "On-board service":
        "On-board service",

    "Leg room service":
        "Leg room",

    "Baggage handling":
        "Baggage handling",

    "Checkin service":
        "Check-in service",

    "Cleanliness":
        "Cleanliness",

    "Online boarding":
        "Online boarding",

    "Departure Delay in Minutes":
        "Departure delay",

    "Arrival Delay in Minutes":
        "Arrival delay",
}


# =========================================================
# FEATURE DESCRIPTIONS
# =========================================================

FEATURE_DESCRIPTIONS = {
    "Seat comfort":
        "Passenger rating of seat comfort.",

    "Departure/Arrival time convenient":
        "Passenger rating of departure and arrival time convenience.",

    "Food and drink":
        "Passenger rating of food and beverage service.",

    "Gate location":
        "Passenger rating of gate location.",

    "Inflight wifi service":
        "Passenger rating of inflight Wi-Fi service.",

    "Inflight entertainment":
        "Passenger rating of inflight entertainment.",

    "Online support":
        "Passenger rating of online customer support.",

    "Ease of Online booking":
        "Passenger rating of online booking convenience.",

    "On-board service":
        "Passenger rating of onboard service.",

    "Leg room service":
        "Passenger rating of available leg room.",

    "Baggage handling":
        "Passenger rating of baggage handling.",

    "Checkin service":
        "Passenger rating of check-in service.",

    "Cleanliness":
        "Passenger rating of cabin cleanliness.",

    "Online boarding":
        "Passenger rating of online boarding.",
}


# =========================================================
# SHAP DISPLAY NAMES
# =========================================================

SHAP_EXPLANATION_NAMES = {
    "Inflight entertainment":
        "Inflight entertainment",

    "Seat comfort":
        "Seat comfort",

    "Customer Type":
        "Customer loyalty",

    "Ease of Online booking":
        "Online booking experience",

    "Leg room service":
        "Leg room",

    "Online support":
        "Online support",

    "Class":
        "Travel class",

    "On-board service":
        "Onboard service",

    "Checkin service":
        "Check-in experience",

    "Online boarding":
        "Online boarding",

    "Baggage handling":
        "Baggage handling",

    "Type of Travel":
        "Travel purpose",

    "Food and drink":
        "Food & drink",

    "Cleanliness":
        "Cabin cleanliness",

    "Departure/Arrival time convenient":
        "Schedule convenience",

    "Gate location":
        "Gate location",

    "Flight Distance":
        "Flight distance",

    "Inflight wifi service":
        "Inflight Wi-Fi",

    "Arrival Delay in Minutes":
        "Arrival delay",

    "Age":
        "Passenger age",

    "Departure Delay in Minutes":
        "Departure delay",
}


# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================

RECOMMENDATION_MAP = {
    "Seat comfort":
        "Review seating comfort, seat condition, and passenger seating experience.",

    "Inflight entertainment":
        "Improve entertainment content variety, system reliability, and available options.",

    "On-board service":
        "Improve onboard service quality, staff responsiveness, and passenger interaction.",

    "Online support":
        "Strengthen digital customer support and reduce response and resolution times.",

    "Ease of Online booking":
        "Simplify the online booking journey and reduce friction during reservations.",

    "Online boarding":
        "Improve the digital boarding experience and make online boarding easier and more reliable.",

    "Inflight wifi service":
        "Improve Wi-Fi availability, connection speed, and service reliability.",

    "Food and drink":
        "Review food and beverage quality, variety, availability, and consistency.",

    "Cleanliness":
        "Increase cabin and aircraft cleanliness standards between and during flights.",

    "Leg room service":
        "Review passenger seating space and improve legroom where operationally feasible.",

    "Baggage handling":
        "Improve baggage handling reliability and reduce baggage-related service issues.",

    "Checkin service":
        "Improve check-in efficiency and reduce passenger waiting time.",

    "Gate location":
        "Improve gate accessibility, signage, and passenger navigation.",

    "Departure/Arrival time convenient":
        "Review schedule convenience and improve operational punctuality.",

    "Departure Delay in Minutes":
        "Reduce departure delays through improved operational planning and turnaround processes.",

    "Arrival Delay in Minutes":
        "Reduce arrival delays through improved flight operations and turnaround efficiency.",
}


# =========================================================
# APPLICATION SETTINGS
# =========================================================

APP_TITLE = (
    "Airline Satisfaction AI"
)


APP_SUBTITLE = (
    "Explainable Machine Learning for predicting passenger satisfaction, "
    "understanding model decisions, and simulating service improvements."
)


# Number of samples used as SHAP background data when
# background-based explanation is required.

SHAP_BACKGROUND_SIZE = 1000


# Maximum number of actionable factors displayed or
# considered in business-priority workflows.

MAX_ACTIONABLE_FACTORS = 5