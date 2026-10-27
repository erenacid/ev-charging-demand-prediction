from pathlib import Path

import pandas as pd


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ev_charging_dataset.csv"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "ev_charging_model_data.csv"
)


# -----------------------------
# Load raw data
# -----------------------------
print("Loading raw EV charging dataset...")

df = pd.read_csv(RAW_DATA_PATH)

print(f"Raw dataset shape: {df.shape}")


# -----------------------------
# Select model features
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


# -----------------------------
# Create modeling dataset
# -----------------------------
model_columns = features + [target]

model_df = df[model_columns].copy()

print("\nSelected columns:")
print(model_df.columns.tolist())

print("\nProcessed dataset shape:")
print(model_df.shape)


# -----------------------------
# Check missing values
# -----------------------------
print("\nMissing values:")
print(model_df.isnull().sum())

total_missing = model_df.isnull().sum().sum()

print(f"\nTotal missing values: {total_missing}")


# -----------------------------
# Identify feature types
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

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# -----------------------------
# Save processed dataset
# -----------------------------
model_df.to_csv(
    PROCESSED_DATA_PATH,
    index=False
)

print(
    f"\nProcessed dataset saved to: "
    f"{PROCESSED_DATA_PATH}"
)

print("\nPreprocessing complete!")