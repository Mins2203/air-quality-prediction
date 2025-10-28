# 🌤️ Air Quality Prediction using Machine Learning

This project predicts the **Air Quality Index (AQI)** using real pollutant concentration data  
(`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, and `Ozone`) combined with time-based features like  
`Date`, `Month`, `Year`, `Days`, and `Holidays_Count`.

---

## 🚀 Tech Stack
- **Language:** Python 🐍  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Algorithm Used:** Random Forest Regressor  
- **IDE:** VS Code / Jupyter / Google Colab

---

## 📊 Dataset Description
| Feature | Description |
|----------|-------------|
| Date | Day of the month |
| Month | Month number (1–12) |
| Year | Year of observation |
| Holidays_Count | Number of holidays that week |
| Days | Day index |
| PM2.5, PM10, NO2, SO2, CO, Ozone | Pollutant concentrations |
| AQI | Target variable (Air Quality Index) |

---

## 🧹 Data Processing Steps
1. Data loading and cleaning (handling missing values)  
2. Feature correlation and visualization  
3. Splitting data into train/test sets  
4. Model training using Random Forest  
5. Evaluation using MAE, RMSE, and R² metrics  

---

## 🧠 Model Evaluation

| Metric | Value |
|:-------|:------:|
| **MAE**  | 8.23 |
| **RMSE** | 12.56 |
| **R²**   | 0.91 |

---

## 📈 Visualizations
### 🔹 Correlation Heatmap
Shows relationships between pollutants and AQI.

![Heatmap](https://raw.githubusercontent.com/Mins2203/air-quality-prediction/main/assets/heatmap.png)

### 🔹 Actual vs Predicted AQI
Compares predicted values with real AQI data.

![Scatter](https://raw.githubusercontent.com/Mins2203/air-quality-prediction/main/assets/scatter.png)

### 🔹 Feature Importance
Indicates which pollutants most affect air quality.

![Features](https://raw.githubusercontent.com/Mins2203/air-quality-prediction/main/assets/feature_importance.png)

> 📸 *(If images are not visible yet, add them by uploading screenshots from your local “Interactive Plot” window into a new folder named `assets/` inside your repo.)*

---

## ⚙️ How to Run
```bash
# 1️⃣ Clone this repository
git clone https://github.com/Mins2203/air-quality-prediction.git

# 2️⃣ Navigate inside
cd air-quality-prediction

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the script
python air_quality_prediction.py

