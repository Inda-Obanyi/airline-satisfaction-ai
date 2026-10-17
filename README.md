
# Airline Satisfaction AI

An end-to-end machine learning application for predicting airline passenger satisfaction using Random Forest, SHAP explainability, and interactive What-If analysis.

## Project Overview

Airline Satisfaction AI predicts whether an airline passenger is satisfied or neutral/dissatisfied based on passenger characteristics and service experience ratings.

The application also provides model explanations and explores how changes to passenger experience inputs affect predicted satisfaction.

## Key Features

- Passenger satisfaction prediction
- Random Forest classification
- 21 model input features
- SHAP explainability
- Interactive What-If simulator
- Service improvement recommendations
- Input validation and error handling
- Automated testing with Pytest
- Interactive Streamlit dashboard

## Dataset

- Dataset size: 129,880 records
- Input features: 21
- Target: Passenger satisfaction
- Classification: Binary

Target classes:

- 0: Neutral/Dissatisfied
- 1: Satisfied

## Model Performance

The deployment candidate uses a Random Forest Classifier with 100 trees.

| Metric | Test Score |
|---|---|
| Accuracy | 95.57% |
| Precision | 96.87% |
| Recall | 94.97% |
| F1 Score | 0.9591 |
| ROC-AUC | 0.9927 |

These metrics are from held-out dataset evaluation and do not guarantee real-world performance.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- Pytest

## Installation

Clone the repository:

```bash
git clone https://github.com/Inda-Obanyi/airline-satisfaction-ai.git
cd airline-satisfaction-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The trained model must be available at:

`models/airline_satisfaction_rf_pipeline.joblib`

## Automated Testing

Run all automated tests:

```bash
python -m pytest tests/ -q
```

The test suite covers model integrity, input validation, prediction consistency, and What-If comparison.

## Explainable AI

SHAP is used to explain individual predictions and identify which features influence the model's outputs.

SHAP contributions describe model behavior rather than causal relationships.

## What-If Simulator

The What-If simulator allows users to modify passenger experience ratings and compare predicted satisfaction probabilities.

These are hypothetical model predictions, not guaranteed real-world outcomes.

## Project Status

Developed prototype.

Automated model and validation tests have been implemented. Additional explainability verification, probability calibration, and deployment checks are planned.

## Author

**Inda Obanyi**

AI/ML Developer  
Yaqini & Duniya Technology

Portfolio: https://indaobanyi.vercel.app

GitHub: https://github.com/Inda-Obanyi

## License

License to be determined.
