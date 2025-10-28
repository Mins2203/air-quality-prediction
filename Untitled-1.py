# --------------------------------------------------------
# 🌤️ AIR QUALITY INDEX (AQI) PREDICTION - Final Version
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# Features and target
X = df[['Date', 'Month', 'Year', 'Holidays_Count', 'Days', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']]
y = df['AQI']

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# -----------------------------
# STEP 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# -----------------------------
# STEP 8: Visualizations
# -----------------------------
# Actual vs Predicted AQI
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="teal", alpha=0.7)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.grid(True)
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind="barh", color="orange", figsize=(8, 5))
plt.title("Feature Importance (Random Forest)")
plt.show()

print("\n🎯 Done! Air Quality Prediction Project executed successfully.")
import matplotlib
matplotlib.use('TkAgg')  # ensure GUI backend
plt.show(block=True)

