# --------------------------------------------------------
# 🌤️ AIR QUALITY INDEX (AQI) PREDICTION - Final Enhanced Version
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# STEP 1: Load the Dataset
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv("data/final_dataset.csv")

print("✅ Dataset loaded successfully!")
print("\nSample rows:")
print(df.head())

# -----------------------------
# STEP 2: Data Cleaning
# -----------------------------
print("\n🧹 Checking for missing values...")
print(df.isnull().sum())

# Drop missing values if any
df = df.dropna()
print(f"\n✅ After cleaning, dataset shape: {df.shape}")

# -----------------------------
# STEP 3: Exploratory Analysis
# -----------------------------
print("\n📊 Summary Statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Air Quality Features")
plt.show()

# -----------------------------
# STEP 4: Feature Selection
# -----------------------------
X = df[['Date', 'Month', 'Year', 'Holidays_Count', 'Days',
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']]
y = df['AQI']

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6A: Train Random Forest
# -----------------------------
print("\n🌳 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# -----------------------------
# STEP 6B: Train XGBoost Model
# -----------------------------
print("\n⚡ Training XGBoost Model...")
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# -----------------------------
# STEP 7: Model Evaluation
# -----------------------------
def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📈 {name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest")
xgb_results = evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Compare results
results_df = pd.DataFrame([rf_results, xgb_results])
print("\n🔍 Model Comparison:")
print(results_df)

# -----------------------------
# STEP 8: Feature Importance
# -----------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind="barh", color="orange", figsize=(8,5))
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("assets/feature_importance.png")
plt.show()

# -----------------------------
# STEP 9: LSTM Time-Series Forecasting
# -----------------------------
print("\n⏳ Building LSTM model for time-series AQI forecasting...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Sort by date if available
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')

scaler = MinMaxScaler()
scaled_aqi = scaler.fit_transform(df[['AQI']])

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_seq, y_seq = create_sequences(scaled_aqi, seq_length)
split = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

# Define LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=16, validation_data=(X_test_seq, y_test_seq), verbose=1)

# Predictions
y_pred_lstm = model.predict(X_test_seq)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_inv = scaler.inverse_transform(y_test_seq)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='Actual AQI', color='blue')
plt.plot(y_pred_lstm, label='Predicted AQI (LSTM)', color='red')
plt.title("LSTM Time-Series AQI Forecasting")
plt.xlabel("Days")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig("assets/lstm_forecast.png")
plt.show()

print("\n🎯 Done! Full Air Quality Prediction Pipeline executed successfully — Random Forest, XGBoost, and LSTM models built and compared.")
