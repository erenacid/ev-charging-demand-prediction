print("Starting prediction script...")

from pathlib import Path

print("Importing packages...")

import joblib
import pandas as pd

print("Packages imported!")

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_charging_demand.joblib"
)


# -----------------------------
# Load trained model
# -----------------------------
print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# -----------------------------
# Load processed dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)


# -----------------------------
# Feature lists
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
# Create ideal/complete input
# -----------------------------
ideal_input = {}

# Use most common value for categorical features
for feature in categorical_features:
    ideal_input[feature] = df[feature].mode()[0]

# Use median value for numerical features
for feature in numerical_features:
    ideal_input[feature] = df[feature].median()


# Convert input into a DataFrame
ideal_df = pd.DataFrame([ideal_input])


# -----------------------------
# Display input
# -----------------------------
print("\n--- IDEAL INPUT DATA ---")

for column in ideal_df.columns:
    print(f"{column}: {ideal_df.iloc[0][column]}")


# -----------------------------
# Make prediction
# -----------------------------
prediction = model.predict(ideal_df)[0]


# -----------------------------
# Display prediction
# -----------------------------
print("\n--- MODEL PREDICTION ---")

