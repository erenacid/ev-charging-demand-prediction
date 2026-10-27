from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev_charging_model_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_charging_demand.joblib"
)


# -----------------------------
# Load processed data
# -----------------------------
print("Loading processed EV charging data...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# -----------------------------
# Define features and target
# -----------------------------
features = [
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
    "time_slot"
]

target = "charging_demand"

X = df[features]
y = df[target]

print(f"\nNumber of input features: {len(features)}")
print(f"Target variable: {target}")


# -----------------------------
# Define feature types
# -----------------------------
categorical_features = [
    "station_id",
    "location_type",
    "vehicle_type",
    "traffic_density",
    "weather_condition",
    "day_of_week",
    "time_slot"
]

numerical_features = [
    "battery_capacity_kWh",
    "initial_soc",
    "queue_length",
    "station_load",
    "electricity_price",
    "renewable_energy_ratio"
]


# -----------------------------
# Split data into training/testing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain/test split complete.")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


# -----------------------------
# Preprocess categorical features
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# -----------------------------
# Create Random Forest model
# -----------------------------
random_forest = RandomForestRegressor(
    n_estimators=25,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# Build ML pipeline
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ]
)


# -----------------------------
# Train model
# -----------------------------
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training complete!")


# -----------------------------
# Generate predictions
# -----------------------------
print("\nGenerating predictions...")

predictions = model.predict(X_test)


# -----------------------------
# Evaluate model
# -----------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n--- MODEL PERFORMANCE ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# -----------------------------
# Show example predictions
# -----------------------------
results = pd.DataFrame({
    "Actual Demand": y_test.values,
    "Predicted Demand": predictions
})

print("\n--- SAMPLE PREDICTIONS ---")
print(results.head(10))

# -----------------------------
# Create ideal input for demo
# -----------------------------
ideal_input = {}

# Use common categorical values
for feature in categorical_features:
