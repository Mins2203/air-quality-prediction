.

🌤️ Air Quality Index (AQI) Prediction

This project predicts the Air Quality Index (AQI) using machine learning and deep learning models — combining Random Forest, XGBoost, and LSTM for time-series forecasting.
It demonstrates a full end-to-end pipeline from data preprocessing to model evaluation and forecast visualization.

📁 Project Overview

Air quality prediction helps policymakers and citizens make informed environmental decisions.
This project uses pollutant data — PM2.5, PM10, NO₂, SO₂, CO, and Ozone — along with temporal features to forecast AQI trends over time.

Goal:

Predict daily AQI values using ML regressors

Compare model performances

Forecast AQI trends using LSTM

⚙️ Features

✅ Data Cleaning & Preprocessing
✅ Exploratory Data Analysis (EDA)
✅ Correlation Heatmap Visualization
✅ Random Forest and XGBoost Regression
✅ LSTM Time-Series Forecasting
✅ Evaluation Metrics (MAE, RMSE, R²)
✅ Feature Importance Visualization

🧠 Tech Stack
Category	Tools / Libraries
Language	Python
ML Models	RandomForestRegressor, XGBRegressor
DL Model	LSTM (TensorFlow/Keras)
Visualization	Matplotlib, Seaborn
Data Handling	Pandas, NumPy
Evaluation Metrics	MAE, RMSE, R²
📊 Dataset Information

File used: data/final_dataset.csv

Columns:
Date, Month, Year, Holidays_Count, Days, PM2.5, PM10, NO2, SO2, CO, Ozone, AQI

Sample Data:

Date	Month	Year	Holidays_Count	Days	PM2.5	PM10	NO2	SO2	CO	Ozone	AQI
1	1	2021	0	5	408.80	442.42	160.61	12.95	2.77	43.19	462
2	1	2021	0	6	404.04	561.95	52.85	5.18	2.60	16.43	482
3	1	2021	1	7	225.07	239.04	170.95	10.93	1.40	44.29	263
4	1	2021	0	1	89.55	132.08	153.98	10.42	1.01	49.19	207
5	1	2021	0	2	54.06	55.54	122.66	9.70	0.64	48.88	149

Data Cleaning:

Checked and removed missing/null values

Converted Date to datetime for temporal sorting

Final cleaned dataset shape: ✅ (as printed in your output)

📊 Exploratory Data Analysis (EDA)

Summary Statistics: Displayed using df.describe()

Correlation Heatmap: Shows feature correlations with AQI
(saved as assets/feature_importance.png)

Example insight:

High correlation observed between PM2.5, PM10, and AQI, confirming their strong impact on air quality levels.

🚀 Model Training & Evaluation
🌳 Random Forest Regressor

Ensemble-based regression model

Parameters: n_estimators=200, random_state=42

Outputs MAE, RMSE, R² metrics

⚡ XGBoost Regressor

Gradient boosting-based model

Parameters:

n_estimators=300
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
random_state=42

⏳ LSTM Time-Series Model

Deep learning model for AQI forecasting

Sequence length = 10 days

Input: Scaled AQI values

Output: Predicted AQI trend

Trained for 25 epochs, batch size = 16

🧮 Evaluation Metrics

Each model is evaluated on the test set using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

**Model	           MAE	     RMSE	     R²
Random Forest  18.78      28.98     0.94
XGBoost	       18.77      27.97     0.94**


The LSTM model’s predictions are visualized in a time-series plot (see below).

📈 Visual Outputs
Visualization	Description
Correlation Heatmap	Feature correlation overview
Feature Importance Plot	Top features affecting AQI (Random Forest)
LSTM Forecast Plot	Predicted vs Actual AQI trends

📂 Saved in:

assets/
 ┣ feature_importance.png
 ┗ lstm_forecast.png

🧩 Folder Structure
📦 Air-Quality-Prediction
 ┣ 📂 data
 ┃ ┗ 📜 final_dataset.csv
 ┣ 📂 assets
 ┃ ┣ 📊 feature_importance.png
 ┃ ┗ 📈 lstm_forecast.png
 ┣ 📜 air_quality_prediction.py
 ┣ 📜 README.md
 ┗ 📜 requirements.txt

🧪 How to Run
1️⃣ Clone the repository
git clone https://github.com/Mins2203/air-quality-prediction.git
cd air-quality-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the project
python air_quality_prediction.py

📦 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow

📚 Key Insights

PM2.5 and PM10 are the strongest contributors to AQI.

XGBoost tends to outperform Random Forest slightly in R² and RMSE.

LSTM captures temporal AQI variations effectively for future trend prediction.


🏁 Conclusion

This project integrates Machine Learning and Deep Learning techniques for accurate AQI prediction and forecasting.
It provides a scalable foundation for AI-powered environmental monitoring systems 🌍
