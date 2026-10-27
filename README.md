# EV Charging Demand Prediction

## Project Overview

This project uses machine learning to predict electric vehicle charging demand. We use a Random Forest regression model and information such as vehicle type, battery capacity, station load, electricity price, weather, and time of day to make predictions.

## Dataset

The project uses the [Electric Vehicle Charging Demand Dataset](https://www.kaggle.com/datasets/ziya07/electric-vehicle-charging-demand-dataset).

The target variable is `charging_demand`. The project includes a processed version of the dataset in `data/processed/`.

## Installation

You will need Python and the following libraries:

- pandas
- scikit-learn
- joblib
- kagglehub

Install the libraries using:

```bash
pip install pandas scikit-learn joblib kagglehub
```

## Getting Started

First, download or clone the repository:

```bash
git clone https://github.com/erenacid/ev-charging-demand-prediction.git
cd ev-charging-demand-prediction
```

Run preprocessing and train the model:

```bash
python src/preprocess.py
python src/train_model.py
```

The training script splits the data into training and testing sets, trains a Random Forest model, and prints evaluation results including MAE, RMSE, and R². It saves the trained model in the `models/` folder.

To run a prediction:

```bash
python src/predict_demo.py
```

The prediction script creates an example using the most common categorical values and median numerical values from the processed dataset.

## Project Files

- `src/ingest_data.py`: Downloads and inspects the dataset.
- `src/preprocess.py`: Selects the features and target used by the model and saves the processed dataset.
- `src/train_model.py`: Trains and evaluates the Random Forest model.
- `src/predict_demo.py`: Loads the saved model and makes an example prediction.
- `data/raw/`: Contains the original dataset.
- `data/processed/`: Contains the processed dataset used for modeling.
- `models/`: Contains the saved model.

## Contact

For questions, contact Aditi Babar at ababar@charlotte.edu.

## License

[Add a link to your group's license file here once the file has been added.]
