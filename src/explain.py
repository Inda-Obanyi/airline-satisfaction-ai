# =========================================================
# Airline Satisfaction AI
# Explainable AI / SHAP Engine
# =========================================================

import numpy as np
import shap

from .config import (
    ACTIONABLE_FEATURES,
    DISPLAY_NAMES,
    RATING_FEATURES,
    RECOMMENDATION_MAP,
    SHAP_EXPLANATION_NAMES,
    SPECIAL_RATING_VALUE,
    RATING_SCALE_MIN,
    RATING_SCALE_MAX,
    DELAY_OPPORTUNITY_CAP_MINUTES,
    format_rating,
)

from .predict import (
    FEATURES,
    model,
    prepare_model_input,
    predict_satisfaction,
)


# =========================================================
# MODEL COMPONENTS
# =========================================================

if not hasattr(model, "named_steps"):
    raise ValueError(
        "Loaded model does not expose pipeline steps."
    )


if "preprocessor" not in model.named_steps:
    raise ValueError(
        "Model pipeline does not contain a "
        "'preprocessor' step."
    )


if "classifier" not in model.named_steps:
    raise ValueError(
        "Model pipeline does not contain a "
        "'classifier' step."
    )


preprocessor = model.named_steps[
    "preprocessor"
]

classifier = model.named_steps[
    "classifier"
]


# =========================================================
# MODEL CLASS VALIDATION
# =========================================================

if not hasattr(
    classifier,
    "classes_",
):
    raise ValueError(
        "Classifier does not expose 'classes_'. "
        "SHAP class mapping requires explicit "
        "class labels."
    )


CLASS_LABELS = [
    int(class_label)
    for class_label
    in classifier.classes_
]


EXPECTED_CLASSES = {
    0,
    1,
}


if set(CLASS_LABELS) != EXPECTED_CLASSES:
    raise ValueError(
        "Unexpected classifier classes detected. "
        f"Expected {sorted(EXPECTED_CLASSES)}, "
        f"but found {CLASS_LABELS}."
    )


SATISFIED_CLASS = 1
DISSATISFIED_CLASS = 0


SATISFIED_CLASS_INDEX = CLASS_LABELS.index(
    SATISFIED_CLASS
)


DISSATISFIED_CLASS_INDEX = CLASS_LABELS.index(
    DISSATISFIED_CLASS
)


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.TreeExplainer(
    classifier
)


# =========================================================
# CATEGORY DISPLAY MAPPINGS
# =========================================================

CATEGORY_MAPPINGS = {
    "Customer Type": {
        0: "Loyal Customer",
        1: "Disloyal Customer",
    },

    "Type of Travel": {
        0: "Business travel",
        1: "Personal Travel",
    },

    "Class": {
        0: "Business",
        1: "Eco",
        2: "Eco Plus",
    },
}


# =========================================================
# FEATURE GROUPS
# =========================================================

DELAY_FEATURES = {
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
}


# =========================================================
# VALUE FORMATTING
# =========================================================

def format_feature_value(
    feature,
    value,
):
    """
    Convert raw model values into human-readable values.

    Rating value 0 is treated as a special
    N/A / not-rated value.
    """

    # -----------------------------------------------------
    # Categorical features
    # -----------------------------------------------------

    if feature in CATEGORY_MAPPINGS:

        try:
            return CATEGORY_MAPPINGS[
                feature
            ].get(
                int(value),
                str(value),
            )

        except (
            ValueError,
            TypeError,
        ):
            return str(value)

    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    if feature == "Age":

        try:
            return f"{int(value)} years"

        except (
            ValueError,
            TypeError,
        ):
            return str(value)

    # -----------------------------------------------------
    # Flight distance
    # -----------------------------------------------------

    if feature == "Flight Distance":

        try:
            return f"{float(value):,.0f} km"

        except (
            ValueError,
            TypeError,
        ):
            return str(value)

    # -----------------------------------------------------
    # Delays
    # -----------------------------------------------------

    if feature in DELAY_FEATURES:

        try:
            return f"{float(value):,.0f} min"

        except (
            ValueError,
            TypeError,
        ):
            return str(value)

    # -----------------------------------------------------
    # Experience ratings
    # -----------------------------------------------------

    if feature in RATING_FEATURES:

        try:
            return format_rating(
                value
            )

        except (
            ValueError,
            TypeError,
        ):
            return str(value)

    # -----------------------------------------------------
    # Default
    # -----------------------------------------------------

    return str(value)


# =========================================================
# IMPROVEMENT OPPORTUNITY
# =========================================================

def get_improvement_opportunity(
    feature,
    value,
):
    """
    Calculate normalized business improvement opportunity.

    Returns a value between 0.0 and 1.0 so that different
    feature units can be compared in the business-priority
    heuristic.

    ---------------------------------------------------------
    EXPERIENCE RATINGS
    ---------------------------------------------------------

        1/5 -> 1.00 opportunity
        2/5 -> 0.75 opportunity
        3/5 -> 0.50 opportunity
        4/5 -> 0.25 opportunity
        5/5 -> 0.00 opportunity

    Rating value 0 is treated as N/A / not-rated
    and therefore returns None.

    ---------------------------------------------------------
    DELAYS
    ---------------------------------------------------------

    Delay opportunity is normalized against the configured
    operational threshold.

    Example with a 180-minute cap:

        0 min     -> 0.00
        30 min    -> 0.17
        60 min    -> 0.33
        120 min   -> 0.67
        180+ min  -> 1.00

    IMPORTANT:

    This is a business prioritization heuristic.

    It is NOT:
        - a causal estimate
        - a probability
        - a model prediction
        - a guarantee of business impact
    """

    if value is None:
        return None

    # -----------------------------------------------------
    # Convert value safely
    # -----------------------------------------------------

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    # =====================================================
    # EXPERIENCE RATINGS
    # =====================================================

    if feature in RATING_FEATURES:

        # -------------------------------------------------
        # 0 means N/A / not rated.
        # -------------------------------------------------

        if value == SPECIAL_RATING_VALUE:
            return None

        # -------------------------------------------------
        # Protect against invalid rating configuration.
        # -------------------------------------------------

        rating_range = (
            RATING_SCALE_MAX
            - RATING_SCALE_MIN
        )

        if rating_range <= 0:
            return 0.0

        # -------------------------------------------------
        # Calculate remaining normalized opportunity.
        # -------------------------------------------------

        opportunity = (
            RATING_SCALE_MAX
            - value
        ) / rating_range

        # -------------------------------------------------
        # Keep opportunity between 0 and 1.
        # -------------------------------------------------

        return float(
            min(
                1.0,
                max(
                    0.0,
                    opportunity,
                ),
            )
        )

    # =====================================================
    # DELAY FEATURES
    # =====================================================

    if feature in DELAY_FEATURES:

        delay_cap = (
            DELAY_OPPORTUNITY_CAP_MINUTES
        )

        if delay_cap <= 0:
            return 0.0

        # -------------------------------------------------
        # Normalize delay against business threshold.
        # -------------------------------------------------

        opportunity = (
            value / delay_cap
        )

        # -------------------------------------------------
        # Cap opportunity at 1.0.
        # -------------------------------------------------

        return float(
            min(
                1.0,
                max(
                    0.0,
                    opportunity,
                ),
            )
        )

    # =====================================================
    # NON-ACTIONABLE FEATURES
    # =====================================================

    return 0.0


# =========================================================
# HUMAN-READABLE IMPROVEMENT DETAILS
# =========================================================

def get_improvement_details(
    feature,
    value,
):
    """
    Return human-readable improvement information.

    This keeps the normalized business-priority score
    separate from the actual business units displayed
    to the user.
    """

    details = {
        "remaining_improvement_points": None,
        "potential_delay_reduction_minutes": None,
    }

    if value is None:
        return details

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return details

    # =====================================================
    # RATINGS
    # =====================================================

    if feature in RATING_FEATURES:

        # 0 = N/A / not rated
        if numeric_value == SPECIAL_RATING_VALUE:
            return details

        details[
            "remaining_improvement_points"
        ] = max(
            0.0,
            RATING_SCALE_MAX
            - numeric_value,
        )

        return details

    # =====================================================
    # DELAYS
    # =====================================================

    if feature in DELAY_FEATURES:

        details[
            "potential_delay_reduction_minutes"
        ] = max(
            0.0,
            numeric_value,
        )

        return details

    return details


# =========================================================
# BUSINESS PRIORITY SCORE
# =========================================================

def calculate_business_priority(
    factor,
):
    """
    Calculate a normalized business priority score.

    Formula:

        Priority Score =
            Absolute SHAP Impact
            ×
            Normalized Improvement Opportunity

    IMPORTANT:

    This is a decision-support heuristic.

    It is NOT:
        - a causal estimate
        - a probability
        - a percentage
        - a model prediction
    """

    # -----------------------------------------------------
    # Extract SHAP impact
    # -----------------------------------------------------

    shap_value = factor.get(
        "shap_value",
        0.0,
    )

    # -----------------------------------------------------
    # Extract feature
    # -----------------------------------------------------

    feature = factor.get(
        "feature"
    )

    # -----------------------------------------------------
    # Extract current value
    # -----------------------------------------------------

    current_value = factor.get(
        "current_value"
    )

    if current_value is None:

        current_value = factor.get(
            "value"
        )

    # -----------------------------------------------------
    # Convert SHAP safely
    # -----------------------------------------------------

    try:

        shap_impact = abs(
            float(shap_value)
        )

    except (
        TypeError,
        ValueError,
    ):

        shap_impact = 0.0

    # -----------------------------------------------------
    # Calculate normalized opportunity
    # -----------------------------------------------------

    improvement_opportunity = (
        get_improvement_opportunity(
            feature,
            current_value,
        )
    )

    # -----------------------------------------------------
    # N/A / not-rated values
    # -----------------------------------------------------

    if improvement_opportunity is None:
        return 0.0

    # -----------------------------------------------------
    # Final heuristic score
    # -----------------------------------------------------

    priority_score = (
        shap_impact
        * improvement_opportunity
    )

    return float(
        priority_score
    )


# =========================================================
# TRANSFORM DATA FOR SHAP
# =========================================================

def transform_for_shap(
    passenger_data,
):
    """
    Prepare passenger data and transform it using the
    trained pipeline preprocessor.
    """

    passenger_data = prepare_model_input(
        passenger_data
    )

    transformed_data = (
        preprocessor.transform(
            passenger_data
        )
    )

    return (
        passenger_data,
        transformed_data,
    )


# =========================================================
# SHAP OUTPUT NORMALIZATION
# =========================================================

def _extract_satisfied_shap_values(
    shap_output,
):
    """
    Extract SHAP values corresponding specifically to
    the Satisfied class.

    The function supports common SHAP output formats:

        3D:
            (samples, features, classes)

        2D:
            (samples, features)

    For 3D output, the class index is resolved from
    classifier.classes_ rather than assuming that index 1
    always represents the Satisfied class.
    """

    values = np.asarray(
        shap_output.values
    )

    # =====================================================
    # 3D SHAP OUTPUT
    # =====================================================

    if values.ndim == 3:

        if values.shape[0] < 1:
            raise ValueError(
                "SHAP returned no sample explanations."
            )

        if (
            values.shape[2]
            != len(CLASS_LABELS)
        ):
            raise ValueError(
                "SHAP class dimension does not match "
                "the classifier class count. "
                f"SHAP classes: {values.shape[2]}, "
                f"classifier classes: "
                f"{len(CLASS_LABELS)}."
            )

        satisfied_values = values[
            0,
            :,
            SATISFIED_CLASS_INDEX,
        ]

    # =====================================================
    # 2D SHAP OUTPUT
    # =====================================================

    elif values.ndim == 2:

        if values.shape[0] < 1:
            raise ValueError(
                "SHAP returned no sample explanations."
            )

        satisfied_values = values[
            0
        ]

    # =====================================================
    # UNSUPPORTED OUTPUT
    # =====================================================

    else:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{values.shape}"
        )

    # =====================================================
    # NUMERIC CONVERSION
    # =====================================================

    satisfied_values = np.asarray(
        satisfied_values,
        dtype=float,
    )

    # =====================================================
    # FEATURE COUNT VALIDATION
    # =====================================================

    if len(
        satisfied_values
    ) != len(FEATURES):

        raise ValueError(
            "SHAP feature count does not match "
            "model feature count. "
            f"SHAP returned "
            f"{len(satisfied_values)} values, "
            f"while the model expects "
            f"{len(FEATURES)} features."
        )

    return satisfied_values


# =========================================================
# GET SHAP VALUES
# =========================================================

def get_satisfied_shap_values(
    passenger_data,
):
    """
    Calculate SHAP values specifically for the
    Satisfied class.

    The trained pipeline contains:

        Median Imputation
        StandardScaler
        Random Forest

    No one-hot encoding is used, so the transformed feature
    count should remain equal to the original feature count.
    """

    (
        passenger_data,
        transformed_data,
    ) = transform_for_shap(
        passenger_data
    )

    # =====================================================
    # CALCULATE SHAP EXPLANATION
    # =====================================================

    shap_output = explainer(
        transformed_data
    )

    # =====================================================
    # EXTRACT SATISFIED CLASS
    # =====================================================

    satisfied_values = (
        _extract_satisfied_shap_values(
            shap_output
        )
    )

    return satisfied_values


# =========================================================
# FEATURE EXPLANATION
# =========================================================

def generate_feature_explanation(
    passenger_data,
):
    """
    Create structured SHAP explanations for a passenger.

    Each factor contains:

        - feature
        - display name
        - current value
        - formatted value
        - SHAP value
        - absolute SHAP value
        - direction
        - actionable status
        - normalized improvement opportunity
        - human-readable improvement details
        - business priority score
    """

    # =====================================================
    # PREPARE PASSENGER DATA
    # =====================================================

    passenger_data = prepare_model_input(
        passenger_data
    )

    # =====================================================
    # CALCULATE SHAP VALUES
    # =====================================================

    shap_values = get_satisfied_shap_values(
        passenger_data
    )

    # =====================================================
    # GET PASSENGER ROW
    # =====================================================

    row = passenger_data.iloc[
        0
    ]

    factors = []

    # =====================================================
    # BUILD FEATURE EXPLANATIONS
    # =====================================================

    for index, feature in enumerate(
        FEATURES
    ):

        # -------------------------------------------------
        # SHAP value
        # -------------------------------------------------

        shap_value = float(
            shap_values[index]
        )

        # -------------------------------------------------
        # Current feature value
        # -------------------------------------------------

        current_value = row[
            feature
        ]

        # -------------------------------------------------
        # Determine whether feature is actionable
        # -------------------------------------------------

        actionable = (
            feature in ACTIONABLE_FEATURES
        )

        # -------------------------------------------------
        # Create base factor
        # -------------------------------------------------

        factor = {
            "feature":
                feature,

            "display_name":
                SHAP_EXPLANATION_NAMES.get(
                    feature,
                    DISPLAY_NAMES.get(
                        feature,
                        feature,
                    ),
                ),

            "value":
                current_value,

            "current_value":
                current_value,

            "formatted_value":
                format_feature_value(
                    feature,
                    current_value,
                ),

            "shap_value":
                shap_value,

            "abs_shap_value":
                abs(shap_value),

            "direction":
                (
                    "positive"
                    if shap_value >= 0
                    else "negative"
                ),

            "actionable":
                actionable,
        }

        # =================================================
        # IMPROVEMENT OPPORTUNITY
        # =================================================

        if actionable:

            normalized_opportunity = (
                get_improvement_opportunity(
                    feature,
                    current_value,
                )
            )

        else:

            normalized_opportunity = 0.0

        factor[
            "improvement_opportunity"
        ] = normalized_opportunity

        # =================================================
        # HUMAN-READABLE IMPROVEMENT DETAILS
        # =================================================

        improvement_details = (
            get_improvement_details(
                feature,
                current_value,
            )
        )

        factor.update(
            improvement_details
        )

        # =================================================
        # BUSINESS PRIORITY SCORE
        # =================================================

        if (
            actionable
            and shap_value < 0
            and normalized_opportunity is not None
            and normalized_opportunity > 0
        ):

            factor[
                "business_priority_score"
            ] = calculate_business_priority(
                factor
            )

            factor[
                "has_improvement_opportunity"
            ] = True

        else:

            factor[
                "business_priority_score"
            ] = 0.0

            factor[
                "has_improvement_opportunity"
            ] = False

        # -------------------------------------------------
        # Add factor
        # -------------------------------------------------

        factors.append(
            factor
        )

    # =====================================================
    # MODEL DRIVER RANKING
    # =====================================================

    factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # POSITIVE FACTORS
    # =====================================================

    positive_factors = [
        factor
        for factor in factors
        if factor["shap_value"] > 0
    ]

    positive_factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # NEGATIVE FACTORS
    # =====================================================

    negative_factors = [
        factor
        for factor in factors
        if factor["shap_value"] < 0
    ]

    negative_factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # ACTIONABLE NEGATIVE FACTORS
    # =====================================================

    actionable_negative_factors = [
        factor
        for factor in negative_factors
        if factor["actionable"]
    ]

    actionable_negative_factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # BUSINESS PRIORITIES
    # =====================================================

    business_priority_factors = [
        factor
        for factor in actionable_negative_factors
        if factor[
            "has_improvement_opportunity"
        ]
    ]

    business_priority_factors.sort(
        key=lambda item:
            item["business_priority_score"],
        reverse=True,
    )

    # =====================================================
    # MONITORING FACTORS
    # =====================================================

    monitoring_factors = [
        factor
        for factor in actionable_negative_factors
        if not factor[
            "has_improvement_opportunity"
        ]
    ]

    monitoring_factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # NON-ACTIONABLE NEGATIVE FACTORS
    # =====================================================

    non_actionable_factors = [
        factor
        for factor in negative_factors
        if not factor["actionable"]
    ]

    non_actionable_factors.sort(
        key=lambda item:
            item["abs_shap_value"],
        reverse=True,
    )

    # =====================================================
    # RETURN STRUCTURED EXPLANATION
    # =====================================================

    return {
        "all_factors":
            factors,

        "positive_factors":
            positive_factors,

        "negative_factors":
            negative_factors,

        "actionable_negative_factors":
            actionable_negative_factors,

        "business_priority_factors":
            business_priority_factors,

        "monitoring_factors":
            monitoring_factors,

        "non_actionable_factors":
            non_actionable_factors,
    }


# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    explanation,
    max_recommendations=5,
):
    """
    Convert genuine business-priority factors into
    business recommendations.

    Only factors with:

        negative SHAP impact
        +
        actionable status
        +
        remaining improvement opportunity

    are included.

    Recommendations are ordered by the normalized
    business-priority score.
    """

    recommendations = []

    business_priorities = explanation.get(
        "business_priority_factors",
        [],
    )

    # =====================================================
    # PROCESS PRIORITY FACTORS
    # =====================================================

    for factor in business_priorities:

        feature = factor[
            "feature"
        ]

        # -------------------------------------------------
        # Skip unsupported recommendations
        # -------------------------------------------------

        if feature not in RECOMMENDATION_MAP:
            continue

        # -------------------------------------------------
        # Human-readable improvement data
        # -------------------------------------------------

        remaining_points = factor.get(
            "remaining_improvement_points"
        )

        delay_reduction = factor.get(
            "potential_delay_reduction_minutes"
        )

        # -------------------------------------------------
        # Build recommendation
        # -------------------------------------------------

        recommendations.append(
            {
                "feature":
                    feature,

                "display_name":
                    factor[
                        "display_name"
                    ],

                "current_value":
                    factor[
                        "formatted_value"
                    ],

                "shap_value":
                    factor[
                        "shap_value"
                    ],

                "business_priority_score":
                    factor.get(
                        "business_priority_score",
                        0.0,
                    ),

                "improvement_opportunity":
                    factor.get(
                        "improvement_opportunity",
                        0.0,
                    ),

                "remaining_improvement_points":
                    remaining_points,

                "potential_delay_reduction_minutes":
                    delay_reduction,

                "recommendation":
                    RECOMMENDATION_MAP[
                        feature
                    ],
            }
        )

        # =================================================
        # RESPECT RECOMMENDATION LIMIT
        # =================================================

        if len(
            recommendations
        ) >= max_recommendations:

            break

    return recommendations


# =========================================================
# COMPLETE PASSENGER EXPLANATION
# =========================================================

def explain_passenger(
    passenger_data,
    max_recommendations=5,
):
    """
    Generate the complete explainable AI result.

    Includes:

        - Model prediction
        - Satisfaction probabilities
        - SHAP values
        - Positive factors
        - Negative factors
        - Actionable negative factors
        - Business priorities
        - Monitoring factors
        - Non-actionable factors
        - Business recommendations
    """

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = predict_satisfaction(
        passenger_data
    )

    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    explanation = generate_feature_explanation(
        passenger_data
    )

    # =====================================================
    # BUSINESS RECOMMENDATIONS
    # =====================================================

    recommendations = generate_recommendations(
        explanation,
        max_recommendations=max_recommendations,
    )

    # =====================================================
    # COMPLETE RESULT
    # =====================================================

    return {
        **prediction,

        "positive_factors":
            explanation[
                "positive_factors"
            ],

        "negative_factors":
            explanation[
                "negative_factors"
            ],

        "actionable_negative_factors":
            explanation[
                "actionable_negative_factors"
            ],

        "business_priority_factors":
            explanation[
                "business_priority_factors"
            ],

        "monitoring_factors":
            explanation[
                "monitoring_factors"
            ],

        "non_actionable_factors":
            explanation[
                "non_actionable_factors"
            ],

        "all_factors":
            explanation[
                "all_factors"
            ],

        "recommendations":
            recommendations,
    }