# =========================================================
# Airline Satisfaction AI
# Explainable Passenger Satisfaction Prediction
# Portfolio-Ready Streamlit Application
# =========================================================

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import (
    CLASS_TYPES,
    CUSTOMER_TYPES,
    DISPLAY_NAMES,
    MAX_ACTIONABLE_FACTORS,
    RATING_FEATURES,
    RECOMMENDATION_MAP,
    TRAVEL_TYPES,
    format_rating,
    validate_age,
    validate_delay,
)

from src.predict import compare_predictions
from src.explain import explain_passenger


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Airline Satisfaction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# APPLICATION CONSTANTS
# =========================================================

from src.predict import model, metadata

MODEL_NAME = metadata["model"]["name"].replace(" Classifier", "")

MODEL_TREES = len(
    model.named_steps["classifier"].estimators_
)

MODEL_ROC_AUC = float(
    metadata["test_metrics"]["roc_auc"]
)

EXPECTED_TREES = metadata["model"]["n_estimators"]

if MODEL_TREES != EXPECTED_TREES:
    raise RuntimeError(
        f"Model integrity failure: loaded model has "
        f"{MODEL_TREES} trees, but metadata expects "
        f"{EXPECTED_TREES}."
    )

RATING_OPTIONS = [0, 1, 2, 3, 4, 5]

EXPERIENCE_FEATURES = [
    "Seat comfort",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]

PROCESS_FEATURES = [
    feature
    for feature in RATING_FEATURES
    if feature not in EXPERIENCE_FEATURES
]


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =========================================================
# GLOBAL CSS
#
# CSS only. No HTML is used to create visible application
# content.
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL PAGE
       ===================================================== */

    .stApp {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0f172a;
    }

    /* =====================================================
       STREAMLIT BORDERED CONTAINERS
       ===================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border-color: #e2e8f0;
        background: #ffffff;
    }

    /* =====================================================
       METRICS
       ===================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }

    /* =====================================================
       SELECTBOXES / INPUTS
       ===================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    /* =====================================================
       EXPANDERS
       ===================================================== */

    div[data-testid="stExpander"] {
        border-color: #e2e8f0;
        border-radius: 14px;
        background: #ffffff;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 9px;
        font-weight: 700;
    }

    /* =====================================================
       ALERTS
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* =====================================================
       DATAFRAME
       ===================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* =====================================================
       DIVIDERS
       ===================================================== */

    hr {
        border-color: #e2e8f0;
    }

    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def build_passenger_data(
    customer_type,
    age,
    travel_type,
    travel_class,
    flight_distance,
    departure_delay,
    arrival_delay,
    ratings,
):
    """
    Build the exact dataframe structure expected
    by the trained model.
    """

    data = {
        "Customer Type": CUSTOMER_TYPES[customer_type],
        "Age": validate_age(age),
        "Type of Travel": TRAVEL_TYPES[travel_type],
        "Class": CLASS_TYPES[travel_class],
        "Flight Distance": float(flight_distance),
        "Departure Delay in Minutes": validate_delay(
            departure_delay
        ),
        "Arrival Delay in Minutes": validate_delay(
            arrival_delay
        ),
    }

    data.update(ratings)

    return pd.DataFrame([data])


def format_percentage(value):
    """
    Convert probability from decimal to percentage.
    """

    return f"{float(value) * 100:.1f}%"


def render_section_heading(
    title,
    description=None,
):
    """
    Render a clean Streamlit-native section heading.
    """

    st.header(title)

    if description:
        st.caption(description)


def probability_chart(
    satisfied_probability,
    dissatisfied_probability,
):
    """
    Create a probability distribution chart.
    """

    labels = [
        "Satisfied",
        "Neutral / Dissatisfied",
    ]

    values = [
        satisfied_probability * 100,
        dissatisfied_probability * 100,
    ]

    fig, ax = plt.subplots(
        figsize=(10, 3.6)
    )

    bars = ax.barh(
        labels,
        values,
        height=0.55,
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel(
        "Probability (%)",
        fontsize=10,
    )

    ax.set_title(
        "Probability of Each Prediction Class",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    for bar, value in zip(
        bars,
        values,
    ):

        if value >= 15:

            ax.text(
                value - 1.5,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
            )

        else:

            ax.text(
                value + 1.5,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()

    return fig


def shap_chart(factors):
    """
    Create horizontal SHAP impact chart.
    """

    if not factors:
        return None

    sorted_factors = sorted(
        factors,
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True,
    )[:10]

    sorted_factors = list(
        reversed(sorted_factors)
    )

    labels = [
        factor["display_name"]
        for factor in sorted_factors
    ]

    values = [
        factor["shap_value"]
        for factor in sorted_factors
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    ax.barh(
        labels,
        values,
        height=0.55,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "SHAP impact on model prediction",
        fontsize=10,
    )

    ax.set_title(
        "Top Model Decision Drivers",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()

    return fig


def create_history_record(result):
    """
    Create a compact prediction-history record.
    """

    return {
        "Prediction": result["label"],
        "Satisfied Probability": format_percentage(
            result["satisfied_probability"]
        ),
        "Neutral / Dissatisfied Probability":
            format_percentage(
                result["dissatisfied_probability"]
            ),
        "Confidence": format_percentage(
            result["confidence"]
        ),
    }


def get_improvement_display(factor):
    """
    Return a human-readable improvement opportunity.

    Rating features:
        Remaining rating points.

    Delay features:
        Potential delay reduction in minutes.

    Other features:
        Not applicable.
    """

    remaining_points = factor.get(
        "remaining_improvement_points"
    )

    delay_reduction = factor.get(
        "potential_delay_reduction_minutes"
    )

    if remaining_points is not None:

        points = float(
            remaining_points
        )

        if points <= 0:
            return "No remaining rating improvement"

        return (
            f"Improvement opportunity: "
            f"{points:g} rating point(s)"
        )

    if delay_reduction is not None:

        minutes = float(
            delay_reduction
        )

        if minutes <= 0:
            return "No remaining delay reduction opportunity"

        return (
            f"Potential reduction: "
            f"{minutes:,.0f} minute(s)"
        )

    return "No improvement opportunity"


def get_improvement_value_for_table(factor):
    """
    Return a readable improvement value for the
    technical explanation table.
    """

    remaining_points = factor.get(
        "remaining_improvement_points"
    )

    delay_reduction = factor.get(
        "potential_delay_reduction_minutes"
    )

    if remaining_points is not None:

        return (
            f"{float(remaining_points):g} "
            "rating point(s)"
        )

    if delay_reduction is not None:

        return (
            f"{float(delay_reduction):,.0f} "
            "minute(s)"
        )

    return "N/A"


# =========================================================
# SIDEBAR
# PASSENGER CONFIGURATION
# =========================================================

with st.sidebar:

    st.title("✈️ Passenger Configuration")

    st.caption(
        "Configure the passenger profile, flight "
        "conditions and operational delays."
    )

    # -----------------------------------------------------
    # DEMOGRAPHICS
    # -----------------------------------------------------

    st.subheader("👤 Demographics")

    customer_type = st.selectbox(
        "Customer Type",
        options=list(
            CUSTOMER_TYPES.keys()
        ),
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=35,
        step=1,
    )

    # -----------------------------------------------------
    # TRAVEL DETAILS
    # -----------------------------------------------------

    st.subheader("🧳 Travel Details")

    travel_type = st.selectbox(
        "Type of Travel",
        options=list(
            TRAVEL_TYPES.keys()
        ),
    )

    travel_class = st.selectbox(
        "Class",
        options=list(
            CLASS_TYPES.keys()
        ),
    )

    flight_distance = st.number_input(
        "Flight Distance (km)",
        min_value=1.0,
        value=1000.0,
        step=50.0,
    )

    # -----------------------------------------------------
    # FLIGHT DELAYS
    # -----------------------------------------------------

    st.subheader("⏱️ Flight Delays")

    departure_delay = st.number_input(
        "Departure Delay (minutes)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    arrival_delay = st.number_input(
        "Arrival Delay (minutes)",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    st.divider()

    st.caption(
        "Experience ratings use 1–5. "
        "A value of 0 represents Not Rated / N/A."
    )


# =========================================================
# HERO / INTRODUCTION
# =========================================================

st.title("✈️ Airline Satisfaction AI")

st.write(
    "Explainable Machine Learning for predicting "
    "passenger satisfaction, understanding model "
    "decisions, and simulating service improvements."
)

st.caption(
    "RANDOM FOREST  •  SHAP  •  WHAT-IF ANALYSIS"
)


# =========================================================
# VALUE PROPOSITION
# =========================================================

value_columns = st.columns(
    4,
    gap="medium",
)

value_cards = [
    (
        "🎯 Predict",
        "Passenger satisfaction",
    ),
    (
        "🔎 Explain",
        "Model decisions with SHAP",
    ),
    (
        "📊 Prioritize",
        "Actionable business factors",
    ),
    (
        "🧪 Simulate",
        "What-if improvements",
    ),
]

for column, (
    title,
    description,
) in zip(
    value_columns,
    value_cards,
):

    with column:

        with st.container(border=True):

            st.subheader(title)

            st.caption(description)


# =========================================================
# PASSENGER EXPERIENCE
# =========================================================

render_section_heading(
    "Passenger Experience",
    "Describe the passenger's experience across "
    "service, comfort and digital touchpoints.",
)

st.info(
    "ℹ️ Rating guide: Use 1–5 for passenger experience "
    "ratings. A value of 0 represents Not Rated / N/A "
    "in the underlying airline survey dataset and is "
    "not interpreted as a 0/5 satisfaction score."
)


ratings = {}


input_columns = st.columns(
    2,
    gap="large",
)


# =========================================================
# IN-FLIGHT EXPERIENCE
# =========================================================

with input_columns[0]:

    with st.container(border=True):

        st.subheader(
            "🛫 In-flight Experience"
        )

        st.caption(
            "Comfort, onboard services and physical "
            "passenger experience."
        )

        st.divider()

        for feature in EXPERIENCE_FEATURES:

            display_name = DISPLAY_NAMES.get(
                feature,
                feature,
            )

            ratings[feature] = st.selectbox(
                display_name,
                options=RATING_OPTIONS,
                index=3,
                format_func=format_rating,
                key=f"rating_{feature}",
            )


# =========================================================
# SERVICE & PROCESS
# =========================================================

with input_columns[1]:

    with st.container(border=True):

        st.subheader(
            "💻 Service & Process"
        )

        st.caption(
            "Digital support, booking, boarding and "
            "journey convenience."
        )

        st.divider()

        for feature in PROCESS_FEATURES:

            display_name = DISPLAY_NAMES.get(
                feature,
                feature,
            )

            ratings[feature] = st.selectbox(
                display_name,
                options=RATING_OPTIONS,
                index=3,
                format_func=format_rating,
                key=f"rating_{feature}",
            )


# =========================================================
# BUILD PASSENGER DATA
# =========================================================

passenger_data = build_passenger_data(
    customer_type=customer_type,
    age=age,
    travel_type=travel_type,
    travel_class=travel_class,
    flight_distance=flight_distance,
    departure_delay=departure_delay,
    arrival_delay=arrival_delay,
    ratings=ratings,
)


# =========================================================
# RUN EXPLANATION ENGINE
# =========================================================

result = explain_passenger(
    passenger_data,
    max_recommendations=MAX_ACTIONABLE_FACTORS,
)


# =========================================================
# MODEL STATUS
# =========================================================

st.success(
    f"Model ready — {MODEL_NAME} • "
    f"{MODEL_TREES} trees • "
    f"ROC-AUC {MODEL_ROC_AUC:.3f}"
)


# =========================================================
# PREDICTION RESULT
# =========================================================

render_section_heading(
    "Prediction Result",
    "The trained Random Forest model evaluates the "
    "passenger profile and produces a probability-based "
    "satisfaction prediction.",
)


result_columns = st.columns(
    [1.35, 1, 1, 1],
    gap="medium",
)


# ---------------------------------------------------------
# PREDICTED CLASS
# ---------------------------------------------------------

with result_columns[0]:

    with st.container(border=True):

        st.caption(
            "PREDICTED CLASS"
        )

        st.subheader(
            result["label"]
        )

        st.caption(
            "Primary model prediction"
        )


# ---------------------------------------------------------
# SATISFACTION PROBABILITY
# ---------------------------------------------------------

with result_columns[1]:

    with st.container(border=True):

        st.caption(
            "SATISFIED PROBABILITY"
        )

        st.metric(
            "Satisfaction",
            format_percentage(
                result["satisfied_probability"]
            ),
        )

        st.caption(
            "Probability of satisfaction"
        )


# ---------------------------------------------------------
# DISSATISFACTION PROBABILITY
# ---------------------------------------------------------

with result_columns[2]:

    with st.container(border=True):

        st.caption(
            "NEUTRAL / DISSATISFIED"
        )

        st.metric(
            "Dissatisfaction",
            format_percentage(
                result["dissatisfied_probability"]
            ),
        )

        st.caption(
            "Probability of dissatisfaction"
        )


# ---------------------------------------------------------
# MODEL CONFIDENCE
# ---------------------------------------------------------

with result_columns[3]:

    with st.container(border=True):

        st.caption(
            "MODEL CONFIDENCE"
        )

        st.metric(
            "Confidence",
            format_percentage(
                result["confidence"]
            ),
        )

        st.caption(
            "Highest predicted class probability"
        )


# =========================================================
# PREDICTION PROBABILITY
# =========================================================

render_section_heading(
    "Prediction Probability",
    "Probability distribution across the two model classes.",
)


with st.container(border=True):

    fig = probability_chart(
        result["satisfied_probability"],
        result["dissatisfied_probability"],
    )

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# =========================================================
# WHY THIS PREDICTION?
# =========================================================

render_section_heading(
    "Why This Prediction?",
    "The strongest positive and negative model drivers "
    "for this individual passenger.",
)


top_positive = result.get(
    "positive_factors",
    [],
)[:3]

top_negative = result.get(
    "negative_factors",
    [],
)[:3]


why_columns = st.columns(
    2,
    gap="large",
)


# ---------------------------------------------------------
# POSITIVE FACTORS
# ---------------------------------------------------------

with why_columns[0]:

    with st.container(border=True):

        st.subheader(
            "📈 Factors supporting satisfaction"
        )

        if top_positive:

            for index, factor in enumerate(
                top_positive
            ):

                st.write(
                    f"**{factor['display_name']}**"
                )

                st.caption(
                    f"{factor['formatted_value']} "
                    f"• SHAP "
                    f"+{factor['shap_value']:.4f}"
                )

                if index < len(top_positive) - 1:
                    st.divider()

        else:

            st.info(
                "No positive model drivers identified."
            )


# ---------------------------------------------------------
# NEGATIVE FACTORS
# ---------------------------------------------------------

with why_columns[1]:

    with st.container(border=True):

        st.subheader(
            "📉 Factors reducing satisfaction"
        )

        if top_negative:

            for index, factor in enumerate(
                top_negative
            ):

                st.write(
                    f"**{factor['display_name']}**"
                )

                st.caption(
                    f"{factor['formatted_value']} "
                    f"• SHAP "
                    f"{factor['shap_value']:.4f}"
                )

                if index < len(top_negative) - 1:
                    st.divider()

        else:

            st.info(
                "No negative model drivers identified."
            )


# =========================================================
# PASSENGER PROFILE
# =========================================================

render_section_heading(
    "Passenger Profile",
    "Summary of the passenger and flight configuration "
    "used for this prediction.",
)


profile_columns_1 = st.columns(
    4,
    gap="medium",
)


with profile_columns_1[0]:

    st.metric(
        "Customer Type",
        customer_type,
    )


with profile_columns_1[1]:

    st.metric(
        "Travel Type",
        travel_type,
    )


with profile_columns_1[2]:

    st.metric(
        "Class",
        travel_class,
    )


with profile_columns_1[3]:

    st.metric(
        "Age",
        f"{age} years",
    )


profile_columns_2 = st.columns(
    3,
    gap="medium",
)


with profile_columns_2[0]:

    st.metric(
        "Flight Distance",
        f"{flight_distance:,.0f} km",
    )


with profile_columns_2[1]:

    st.metric(
        "Departure Delay",
        f"{departure_delay:,.0f} min",
    )


with profile_columns_2[2]:

    st.metric(
        "Arrival Delay",
        f"{arrival_delay:,.0f} min",
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

render_section_heading(
    "Explainable AI",
    "SHAP shows how individual features influenced this "
    "specific prediction. Positive values push the "
    "prediction toward satisfaction; negative values "
    "push it away.",
)


with st.container(border=True):

    fig = shap_chart(
        result.get(
            "all_factors",
            [],
        )
    )

    if fig is not None:

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

    else:

        st.info(
            "No SHAP explanation is available for "
            "this prediction."
        )

    st.warning(
        "Important: SHAP describes model behavior. "
        "It does not establish that changing a feature "
        "will causally produce the predicted outcome."
    )


# =========================================================
# BUSINESS IMPROVEMENT PRIORITIES
# =========================================================

render_section_heading(
    "Business Improvement Priorities",
    "Actionable negative factors ranked using a "
    "business-priority heuristic combining model impact "
    "and normalized improvement opportunity.",
)

st.caption(
    "Priority Score = absolute SHAP impact × normalized "
    "improvement opportunity. This is a decision-support "
    "heuristic, not a causal measure."
)


business_priorities = result.get(
    "business_priority_factors",
    [],
)

business_priorities = business_priorities[
    :MAX_ACTIONABLE_FACTORS
]


if business_priorities:

    for index, factor in enumerate(
        business_priorities,
        start=1,
    ):

        feature = factor["feature"]

        display_name = factor[
            "display_name"
        ]

        priority_score = float(
            factor.get(
                "business_priority_score",
                0.0,
            )
        )

        recommendation = RECOMMENDATION_MAP.get(
            feature
        )

        improvement_display = (
            get_improvement_display(
                factor
            )
        )

        with st.container(border=True):

            st.subheader(
                f"{index}. {display_name}"
            )

            priority_columns = st.columns(
                3,
                gap="medium",
            )

            with priority_columns[0]:

                st.metric(
                    "Current",
                    factor["formatted_value"],
                )

            with priority_columns[1]:

                st.metric(
                    "SHAP Impact",
                    f"{abs(factor['shap_value']):.4f}",
                )

            with priority_columns[2]:

                st.metric(
                    "Priority Score",
                    f"{priority_score:.4f}",
                )

            st.caption(
                improvement_display
            )

            if recommendation:

                st.info(
                    f"Recommended action: "
                    f"{recommendation}"
                )

else:

    st.success(
        "No immediate business improvement priority "
        "was identified for this passenger."
    )


# =========================================================
# FACTORS TO MONITOR
# =========================================================

monitoring_factors = result.get(
    "monitoring_factors",
    [],
)


if monitoring_factors:

    render_section_heading(
        "Factors to Monitor",
        "These features negatively influenced the model "
        "but currently have no remaining improvement "
        "opportunity under the business rules.",
    )

    monitor_columns = st.columns(
        2,
        gap="medium",
    )

    for index, factor in enumerate(
        monitoring_factors
    ):

        with monitor_columns[
            index % 2
        ]:

            with st.container(border=True):

                st.write(
                    f"**{factor['display_name']}**"
                )

                st.caption(
                    f"{factor['formatted_value']} "
                    f"• SHAP "
                    f"{factor['shap_value']:.4f}"
                )


# =========================================================
# WHAT-IF IMPROVEMENT SIMULATOR
# =========================================================

render_section_heading(
    "What-If Improvement Simulator",
    "Test how the trained model responds when selected "
    "business factors are hypothetically improved.",
)


st.info(
    "This is a model simulation, not a causal forecast. "
    "It shows how the trained model responds to changed "
    "inputs; it does not guarantee that the same change "
    "will occur in the real world."
)


what_if_data = passenger_data.copy()

selected_factors = business_priorities[
    :MAX_ACTIONABLE_FACTORS
]


if selected_factors:

    st.subheader(
        "🎯 Simulate Improvements"
    )

    for index, factor in enumerate(
        selected_factors
    ):

        feature = factor["feature"]

        current_value = float(
            passenger_data.iloc[0][feature]
        )

        display_name = DISPLAY_NAMES.get(
            feature,
            feature,
        )

        # -------------------------------------------------
        # RATING FEATURE
        # -------------------------------------------------

        if feature in RATING_FEATURES:

            # 0 means N/A / Not Rated.
            # It is not treated as a poor rating.

            if current_value == 0:

                st.info(
                    f"{display_name} is Not Rated / N/A "
                    "and is excluded from improvement "
                    "simulation."
                )

                continue

            minimum_value = int(
                current_value
            )

            maximum_value = 5

            if minimum_value >= maximum_value:

                simulated_value = minimum_value

                st.write(
                    f"**{display_name}** is already at "
                    f"{format_rating(minimum_value)}."
                )

            else:

                simulated_options = list(
                    range(
                        minimum_value,
                        maximum_value + 1,
                    )
                )

                simulated_value = st.selectbox(
                    f"Improve {display_name}",
                    options=simulated_options,
                    index=len(
                        simulated_options
                    ) - 1,
                    format_func=format_rating,
                    key=f"what_if_rating_{index}",
                )

            st.caption(
                f"Current: "
                f"{format_rating(minimum_value)}"
                f"  →  "
                f"Simulated: "
                f"{format_rating(simulated_value)}"
            )

        # -------------------------------------------------
        # DELAY FEATURE
        # -------------------------------------------------

        elif "Delay" in feature:

            maximum_value = int(
                current_value
            )

            if maximum_value <= 0:

                simulated_value = 0

                st.write(
                    f"**{display_name}** is already "
                    "at 0 minutes."
                )

            else:

                simulated_value = st.number_input(
                    f"Reduce {display_name}",
                    min_value=0,
                    max_value=maximum_value,
                    value=0,
                    step=1,
                    format="%d",
                    key=f"what_if_delay_{index}",
                )

            st.caption(
                f"Current: "
                f"{current_value:.0f} min"
                f"  →  "
                f"Simulated: "
                f"{simulated_value:.0f} min"
            )

        else:

            continue

        what_if_data.loc[
            0,
            feature,
        ] = simulated_value


    # =====================================================
    # BASELINE VS SIMULATED
    # =====================================================

    st.subheader(
        "Baseline vs Simulated Scenario"
    )

    comparison = compare_predictions(
        passenger_data,
        what_if_data,
    )

    before_probability = comparison[
        "baseline_probability"
    ]

    after_probability = comparison[
        "what_if_probability"
    ]

    change = comparison[
        "change_percentage_points"
    ]

    comparison_columns = st.columns(
        3,
        gap="medium",
    )

    with comparison_columns[0]:

        st.metric(
            "Baseline",
            f"{before_probability * 100:.1f}%",
        )

    with comparison_columns[1]:

        st.metric(
            "Simulated",
            f"{after_probability * 100:.1f}%",
        )

    with comparison_columns[2]:

        st.metric(
            "Change",
            f"{change:+.1f} pp",
        )

    baseline_result = comparison[
        "baseline"
    ]

    simulated_result = comparison[
        "what_if"
    ]

    baseline_prediction = baseline_result[
        "prediction"
    ]

    simulated_prediction = simulated_result[
        "prediction"
    ]

    # -----------------------------------------------------
    # CLASS CHANGE INTERPRETATION
    # -----------------------------------------------------

    if (
        baseline_prediction == 0
        and simulated_prediction == 1
    ):

        st.success(
            "The simulated scenario changes the "
            "model prediction from "
            "NEUTRAL / DISSATISFIED to SATISFIED."
        )

    elif (
        baseline_prediction == 1
        and simulated_prediction == 0
    ):

        st.warning(
            "The simulated scenario changes the "
            "model prediction from SATISFIED to "
            "NEUTRAL / DISSATISFIED."
        )

    elif simulated_prediction == 1:

        st.success(
            "The simulated scenario keeps the model "
            "prediction at SATISFIED."
        )

    else:

        st.warning(
            "The simulated scenario keeps the model "
            "prediction at NEUTRAL / DISSATISFIED."
        )

    # -----------------------------------------------------
    # PROBABILITY CHANGE INTERPRETATION
    # -----------------------------------------------------

    if change > 0:

        st.success(
            f"The simulated scenario increases the "
            f"model's satisfaction probability by "
            f"{change:.1f} percentage points."
        )

    elif change < 0:

        st.warning(
            f"The simulated scenario decreases the "
            f"model's satisfaction probability by "
            f"{abs(change):.1f} percentage points."
        )

    else:

        st.info(
            "The simulated scenario does not change "
            "the model's satisfaction probability."
        )

else:

    st.info(
        "No improvement opportunities are currently "
        "available for simulation."
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

render_section_heading(
    "Business Insights",
    "A compact decision-support summary generated from "
    "the current model explanation.",
)


business_priority_count = len(
    business_priorities
)

monitoring_count = len(
    monitoring_factors
)

recommendation_count = len(
    result.get(
        "recommendations",
        [],
    )
)


if business_priorities:

    top_priority = business_priorities[0][
        "display_name"
    ]

else:

    top_priority = "None identified"


dashboard_columns = st.columns(
    4,
    gap="medium",
)


with dashboard_columns[0]:

    st.metric(
        "Satisfaction Probability",
        format_percentage(
            result["satisfied_probability"]
        ),
    )


with dashboard_columns[1]:

    st.metric(
        "Top Priority",
        top_priority,
    )


with dashboard_columns[2]:

    st.metric(
        "Factors to Monitor",
        monitoring_count,
    )


with dashboard_columns[3]:

    st.metric(
        "Recommendations",
        recommendation_count,
    )


# =========================================================
# EXECUTIVE INTERPRETATION
# =========================================================

render_section_heading(
    "Executive Interpretation",
    "A concise business interpretation of the "
    "current model result.",
)


if result["prediction"] == 1:

    interpretation = (
        "The model currently predicts this passenger "
        "as satisfied. The recommended business "
        "approach is to protect the strongest positive "
        "experience drivers while monitoring any "
        "negative model contributors."
    )

else:

    interpretation = (
        "The model currently predicts this passenger "
        "as neutral/dissatisfied. Business attention "
        "should focus on the highest-ranked actionable "
        "factors, particularly those combining strong "
        "negative model impact with substantial room "
        "for improvement."
    )


with st.container(border=True):

    st.subheader(
        "📊 Decision Summary"
    )

    st.write(
        interpretation
    )


# =========================================================
# RECOMMENDED BUSINESS ACTIONS
# =========================================================

render_section_heading(
    "Recommended Business Actions",
    "Business recommendations generated from the "
    "highest-ranked actionable model factors.",
)


recommendations = result.get(
    "recommendations",
    [],
)


if recommendations:

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        with st.container(border=True):

            action_columns = st.columns(
                [0.08, 0.72, 0.20],
                gap="medium",
            )

            with action_columns[0]:

                st.subheader(
                    str(index)
                )

            with action_columns[1]:

                st.write(
                    f"**{recommendation['display_name']}**"
                )

                st.write(
                    recommendation[
                        "recommendation"
                    ]
                )

                # Human-readable opportunity details
                # are intentionally separated from the
                # normalized business-priority score.

                recommendation_points = (
                    recommendation.get(
                        "remaining_improvement_points"
                    )
                )

                recommendation_delay = (
                    recommendation.get(
                        "potential_delay_reduction_minutes"
                    )
                )

                if recommendation_points is not None:

                    st.caption(
                        f"Remaining improvement: "
                        f"{float(recommendation_points):g} "
                        "rating point(s)"
                    )

                elif recommendation_delay is not None:

                    st.caption(
                        f"Potential reduction: "
                        f"{float(recommendation_delay):,.0f} "
                        "minute(s)"
                    )

            with action_columns[2]:

                st.metric(
                    "Priority",
                    f"{recommendation.get(
                        'business_priority_score',
                        0
                    ):.4f}",
                )

else:

    st.info(
        "No immediate business recommendations "
        "were generated."
    )


# =========================================================
# TECHNICAL MODEL EXPLANATION
# =========================================================

render_section_heading(
    "Technical Model Explanation",
    "Inspect the complete feature-level explanation "
    "behind the current prediction.",
)


with st.expander(
    "🔍 View complete feature-level explanation"
):

    explanation_rows = []

    for factor in result.get(
        "all_factors",
        [],
    ):

        shap_value = float(
            factor["shap_value"]
        )

        if shap_value > 0:

            direction = "Positive"

        elif shap_value < 0:

            direction = "Negative"

        else:

            direction = "Neutral"

        explanation_rows.append(
            {
                "Feature":
                    factor["display_name"],

                "Passenger Value":
                    factor["formatted_value"],

                "SHAP Impact":
                    round(
                        shap_value,
                        4,
                    ),

                "Direction":
                    direction,

                "Actionable":
                    (
                        "Yes"
                        if factor.get(
                            "actionable",
                            False,
                        )
                        else "No"
                    ),

                "Improvement Opportunity":
                    get_improvement_value_for_table(
                        factor
                    ),

                "Business Priority Score":
                    round(
                        float(
                            factor.get(
                                "business_priority_score",
                                0.0,
                            )
                        ),
                        4,
                    ),
            }
        )

    if explanation_rows:

        explanation_table = pd.DataFrame(
            explanation_rows
        )

        st.dataframe(
            explanation_table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No feature-level explanation is available."
        )


# =========================================================
# MODEL & TECHNOLOGY STACK
# =========================================================

render_section_heading(
    "Model & Technology Stack",
    "Tools and techniques used to build the "
    "Airline Satisfaction AI system.",
)


tech_columns = st.columns(
    4,
    gap="medium",
)


technologies = [
    ("🐍", "Python"),
    ("🐼", "Pandas"),
    ("🧠", "Scikit-learn"),
    ("🌲", "Random Forest"),
    ("🔎", "SHAP"),
    ("🎈", "Streamlit"),
    ("💡", "Explainable AI"),
    ("📊", "Business Analytics"),
]


for index, (
    icon,
    technology,
) in enumerate(
    technologies
):

    with tech_columns[
        index % 4
    ]:

        with st.container(border=True):

            st.write(
                f"{icon} **{technology}**"
            )


# =========================================================
# PREDICTION HISTORY
# =========================================================

render_section_heading(
    "Prediction History",
    "Save prediction snapshots during the current "
    "Streamlit session for comparison.",
)


history_record = create_history_record(
    result
)


if st.button(
    "💾 Save Current Prediction",
    use_container_width=True,
):

    st.session_state.prediction_history.append(
        history_record
    )

    st.success(
        "Prediction saved to this session."
    )


if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No predictions have been saved yet."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Airline Satisfaction AI"
)

st.caption(
    "Explainable Machine Learning • Random Forest • "
    "SHAP • What-If Simulation • Business Decision Support"
)

st.caption(
    "Built as an applied machine learning portfolio project."
)