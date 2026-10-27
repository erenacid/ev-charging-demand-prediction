from pathlib import Path

import kagglehub
import pandas as pd


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Create the raw-data folder if it does not already exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Download dataset from Kaggle
# -----------------------------
dataset_handle = "ziya07/electric-vehicle-charging-demand-dataset"

print("Downloading EV charging dataset from Kaggle...")

dataset_path = kagglehub.dataset_download(
    dataset_handle,
    output_dir=str(RAW_DATA_DIR)
)

print(f"Dataset downloaded to: {dataset_path}")


# -----------------------------
# Find CSV files
# -----------------------------
csv_files = list(RAW_DATA_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file was found in the downloaded dataset.")

print("\nCSV files found:")

for file in csv_files:
    print(f"- {file.name}")


# Use the first CSV file found
csv_path = csv_files[0]

print(f"\nLoading: {csv_path.name}")


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(csv_path)


# -----------------------------
# Basic inspection
# -----------------------------
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- TARGET CHECK ---")

if "charging_demand" in df.columns:
    print("charging_demand was found.")
    print(df["charging_demand"].describe())
else:
    print("WARNING: charging_demand was not found.")
    print("We need to inspect the actual column names before continuing.")