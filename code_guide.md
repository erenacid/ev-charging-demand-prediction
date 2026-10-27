EV Charging Demand Prediction: Code Guide
This guide explains what each script in our project does and how the scripts work together. The code is currently organized into scripts rather than separate custom functions.
Project Data
Our project uses an electric vehicle charging dataset to predict charging_demand. The model uses 13 features including vehicle type, battery capacity, station load, electricity price, weather, and time of day.

The sample file data/processed/ev_charging_sample.csv contains 100 rows from our processed dataset. It includes all 13 features and the target column. It can be used to try out model training quickly, but results from such a small sample should not be used to judge the model's overall performance.
Data Collection: src/ingest_data.py
Purpose: Download the dataset and check what is in it.

Input: The Electric Vehicle Charging Demand Dataset from Kaggle.

How it works: The script uses kagglehub to download the data into the data/raw folder. It looks for CSV files and reads the first one it finds. It then prints the number of rows and columns, the column names, a preview, data types, and missing values. It also checks whether charging_demand is included.

Output: The downloaded dataset and information printed in the terminal. The script does not rename the file. The preprocessing script expects the raw file to be named ev_charging_dataset.csv.
Preprocessing: src/preprocess.py
Purpose: Prepare the data for model training.

Input: data/raw/ev_charging_dataset.csv.

How it works: The script loads the raw CSV and selects the 13 features along with the charging_demand target. It checks how many values are missing and displays which features are categorical and numerical. It reports missing values but does not fill them in or remove them. The categorical features are encoded later during training.

Output: data/processed/ev_charging_model_data.csv.
Model Training: src/train_model.py
Purpose: Train and evaluate a model that predicts charging demand.

Input: data/processed/ev_charging_model_data.csv.

How it works: The script separates the features from the target. It uses 80% of the data for training and 20% for testing. The random seed is set to 42 so the split can be repeated.

The script converts categorical values into numbers using one-hot encoding. It passes the numerical features through as they are. These steps are combined with a Random Forest regression model in a pipeline. The model uses 25 trees and learns from the training data. It then predicts charging demand for the test data.

Output: The script is intended to print MAE, RMSE, R², and examples of actual and predicted demand. These measures help us see how well the model performed on the test data.

Current fix needed: The uploaded script ends with an unfinished for loop, so it will not run yet. Remove the unfinished block starting at # Create ideal input for demo and replace it with:
joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to: {MODEL_PATH}")
This saves the trained pipeline as models/random_forest_charging_demand.joblib. It will replace an existing model with the same name.
Prediction Demo: src/predict_demo.py
Purpose: Use the saved model to make one example prediction.

Inputs: The saved model in models/ and the processed CSV in data/processed/.

How it works: The script loads the saved model and processed data. It creates an example using the most common value for each categorical feature and the median value for each numerical feature. It then uses the model to predict charging demand for that example.

Output: The script prints the example inputs and calculates a prediction. In the uploaded version, the prediction is not printed yet. Add this line at the end to display it:
print(f"Predicted charging demand: {prediction:.2f}")
Testing with the Sample Data
1. Place ev_charging_sample.csv in the data/processed/ folder.
2. Fix the unfinished training code as shown above.
3. In src/train_model.py, change the filename in DATA_PATH from ev_charging_model_data.csv to ev_charging_sample.csv.
4. From the project folder, install the required libraries and run the training script:
pip install pandas scikit-learn joblib
python src/train_model.py
5. Change the filename back afterward if you want to train with the full dataset again. Back up your existing saved model before running the script if you want to keep it because the save step replaces the model file.
The prediction script still reads the full processed CSV unless you change its path too.
