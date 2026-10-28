
import os
import warnings
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

app = Flask(__name__)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "google_revenue_prediction_dataset_8000.csv"
)

TARGET = "Total_Revenue_Million_USD"

# Example exchange rate; update when required.
USD_TO_INR_RATE = 88.50

FEATURES = [
    "Search_Queries_Billion",
    "YouTube_Users_Million",
    "Google_Cloud_Revenue_Million_USD",
    "Advertising_Revenue_Million_USD",
    "Employee_Strength",
    "R&D_Expenditure_Million_USD",
    "AI_Investment_Million_USD",
    "Operating_Expenses_Million_USD",
    "Market_Share_Percent",
    "Economic_Growth_Percent",
    "Digital_Engagement_Index"
]


# --------------------------------------------------
# USER-FRIENDLY FEATURE LABELS
# --------------------------------------------------

FEATURE_INFO = [
    {
        "name": "Search_Queries_Billion",
        "label": "Search Queries",
        "unit": "Billion",
        "step": "any"
    },
    {
        "name": "YouTube_Users_Million",
        "label": "YouTube Users",
        "unit": "Million",
        "step": "any"
    },
    {
        "name": "Google_Cloud_Revenue_Million_USD",
        "label": "Google Cloud Revenue",
        "unit": "Million USD",
        "step": "any"
    },
    {
        "name": "Advertising_Revenue_Million_USD",
        "label": "Advertising Revenue",
        "unit": "Million USD",
        "step": "any"
    },
    {
        "name": "Employee_Strength",
        "label": "Employee Strength",
        "unit": "Employees",
        "step": "1"
    },
    {
        "name": "R&D_Expenditure_Million_USD",
        "label": "R&D Expenditure",
        "unit": "Million USD",
        "step": "any"
    },
    {
        "name": "AI_Investment_Million_USD",
        "label": "AI Investment",
        "unit": "Million USD",
        "step": "any"
    },
    {
        "name": "Operating_Expenses_Million_USD",
        "label": "Operating Expenses",
        "unit": "Million USD",
        "step": "any"
    },
    {
        "name": "Market_Share_Percent",
        "label": "Market Share",
        "unit": "Percent",
        "step": "any"
    },
    {
        "name": "Economic_Growth_Percent",
        "label": "Economic Growth",
        "unit": "Percent",
        "step": "any"
    },
    {
        "name": "Digital_Engagement_Index",
        "label": "Digital Engagement Index",
        "unit": "Index",
        "step": "any"
    }
]


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset not found. Put "
        "'google_revenue_prediction_dataset_8000.csv' "
        "in the same folder as app.py.\nExpected path: "
        + DATASET_PATH
    )

df = pd.read_csv(DATASET_PATH)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

# Convert required columns to numeric values
df[required_columns] = df[required_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()
df = df.dropna(subset=required_columns)

if len(df) < 10:
    raise ValueError(
        "Not enough valid rows to train the models."
    )

X = df[FEATURES]
y = df[TARGET]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Default values for prediction form
default_values = X_train.median().to_dict()


# --------------------------------------------------
# SIX MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=5
        ))
    ]),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# TRAIN AND EVALUATE SIX MODELS
# --------------------------------------------------

trained_models = {}
model_results = []

print("\nTraining six revenue prediction models...")

for name, current_model in models.items():

    print(f"Training {name}...")

    current_model.fit(X_train, y_train)

    trained_models[name] = current_model

    test_predictions = current_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    mse = mean_squared_error(
        y_test,
        test_predictions
    )

    rmse = float(np.sqrt(mse))

    r2 = r2_score(
        y_test,
        test_predictions
    )

    model_results.append({
        "name": name,
        "mae": round(float(mae), 2),
        "mse": round(float(mse), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2), 4)
    })


# --------------------------------------------------
# SELECT THE BEST MODEL
# --------------------------------------------------

# Select the model with the highest test R² score.
model_results.sort(
    key=lambda item: item["r2"],
    reverse=True
)

best_model_name = model_results[0]["name"]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)
print("R² score:", model_results[0]["r2"])
print("Flask application is ready.")


# --------------------------------------------------
# SHARED TEMPLATE DATA
# --------------------------------------------------

def render_page(
    form_values=None,
    prediction=None,
    prediction_inr_million=None,
    prediction_inr_crore=None,
    error=None
):

    if form_values is None:
        form_values = default_values.copy()

    return render_template(
        "index.html",
        features=FEATURE_INFO,
        models=model_results,
        best_model=best_model_name,
        best_score=model_results[0]["r2"],
        exchange_rate=USD_TO_INR_RATE,
        form_values=form_values,
        prediction=prediction,
        prediction_inr_million=prediction_inr_million,
        prediction_inr_crore=prediction_inr_crore,
        error=error
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_page()


# --------------------------------------------------
# REVENUE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    form_values = {}

    try:

        input_values = []

        for feature in FEATURES:

            raw_value = request.form.get(
                feature,
                ""
            ).strip()

            if not raw_value:
                raise ValueError(
                    f"Please enter {feature.replace('_', ' ')}."
                )

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(
                    "Please enter valid finite numbers."
                )

            input_values.append(value)
            form_values[feature] = value

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        # Predict revenue using the best trained model
        predicted_revenue = float(
            best_model.predict(input_df)[0]
        )

        if not np.isfinite(predicted_revenue):
            raise ValueError(
                "The model did not produce a valid prediction."
            )

        # Revenue is measured in million USD
        revenue_inr_million = (
            predicted_revenue * USD_TO_INR_RATE
        )

        # 1 crore INR = 10 million INR
        revenue_inr_crore = revenue_inr_million / 10

        return render_page(
            form_values=form_values,
            prediction=predicted_revenue,
            prediction_inr_million=revenue_inr_million,
            prediction_inr_crore=revenue_inr_crore
        )

    except (ValueError, TypeError) as exc:

        return render_page(
            form_values=form_values,
            error=str(exc)
        )

    except Exception:

        app.logger.exception("Prediction failed")

        return render_page(
            form_values=form_values,
            error=(
                "Prediction failed. Check your input values "
                "and try again."
            )
        )


# --------------------------------------------------
# START FLASK SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
