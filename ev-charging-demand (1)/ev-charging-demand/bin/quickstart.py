
#!/usr/bin/env python3
"""
EV Charging Demand Prediction - Quickstart Demo

Loads a small EV charging dataset, preprocesses the features,
trains a Random Forest model, evaluates predictions,
and saves the trained model.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ev_charging_dataset.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "demo_ev_charging_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demo_random_forest.joblib"

# --------------------------------------------------
# Dataset features
# --------------------------------------------------

FEATURES = [
    "station_id",
    "location_type",
    "vehicle_type",
    "battery_capacity_kWh",
    "initial_soc",
    "queue_length",
    "station_load",
    "electricity_price",
    "renewable_energy_ratio",
    "traffic_density",
    "weather_condition",
    "day_of_week",
    "time_slot",
]

TARGET = "charging_demand"

CATEGORICAL_FEATURES = [
    "station_id",
    "location_type",
    "vehicle_type",
    "traffic_density",
    "weather_condition",
    "day_of_week",
    "time_slot",
]

NUMERICAL_FEATURES = [
    "battery_capacity_kWh",
    "initial_soc",
    "queue_length",
    "station_load",
    "electricity_price",
    "renewable_energy_ratio",
]


# --------------------------------------------------
# Step 1: Prepare demo dataset
# --------------------------------------------------

def build_demo_sample():
    """Use an existing demo CSV or create an 80-row sample."""

    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SAMPLE_DATA_PATH.exists():
        print(f"Using existing demo dataset: {SAMPLE_DATA_PATH}")
        return

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {RAW_DATA_PATH}\n"
            "Please add a demo CSV or the original dataset."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Randomly select up to 80 records for the demo.
    sample_size = min(80, len(df))

    sample = df[required_columns].sample(
        n=sample_size,
        random_state=42
    ).copy()

    sample.to_csv(SAMPLE_DATA_PATH, index=False)

    print(f"Created demo dataset: {SAMPLE_DATA_PATH}")
    print(f"Sample shape: {sample.shape}")


# --------------------------------------------------
# Step 2: Train and evaluate model
# --------------------------------------------------

def train_demo_model():
    """Train a Random Forest model using the demo dataset."""

    df = pd.read_csv(SAMPLE_DATA_PATH)

    # Validate required columns.
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove incomplete rows for this lightweight demo.
    df = df[required_columns].dropna().copy()

    if len(df) < 4:
        raise ValueError(
            "At least 4 complete data rows are required."
        )

    if len(df) < 20:
        print(
            "\nWARNING: The demo dataset contains fewer than "
            "20 rows. Evaluation metrics may not be reliable."
        )

    print("\n--- Dataset Information ---")
    print(f"Total rows: {len(df)}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Prediction target: {TARGET}")

    # Separate features and target.
    X = df[FEATURES]
    y = pd.to_numeric(df[TARGET], errors="raise")

    # Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # Encode categorical features and preserve numeric features.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            ),
        ]
    )

    # Build the Random Forest pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=25,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    # Train model.
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    # Generate predictions.
    predictions = model.predict(X_test)

    # Calculate evaluation metrics.
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save trained model.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Display results.
    print("\n--- Quickstart Model Summary ---")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\n--- Example Predictions ---")

    results = pd.DataFrame({
        "Actual Demand": y_test.to_numpy(),
        "Predicted Demand": predictions
    })

    print(results.head(5).to_string(index=False))

    print(f"\nSaved model to: {MODEL_PATH}")
    print("\nQuickstart demo completed successfully!")


# --------------------------------------------------
# Run quickstart
# --------------------------------------------------

if __name__ == "__main__":
    build_demo_sample()
    train_demo_model()

