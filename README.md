# 🗽 NYC Taxi Fare Prediction 🚕  
A machine learning project to predict New York City taxi fares using engineered features and multiple regression models. Built as part of the Kaggle Playground competition.

---

## 🎯 Problem Statement  
Predict the `fare_amount` given pickup and dropoff locations and pickup datetime. The competition evaluates predictions using **Root Mean Squared Error (RMSE)**.

- **Kaggle Link**: [NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)

---

## 📁 Dataset  
Files used from the competition:
- `train.csv` (first 100,000 rows used)
- `test.csv`
- `sample_submission.csv`

Engineered Features:
- `total_distance` using the **Haversine formula**
- `Year`, `Month`, `Day`, `Hours`, `Minutes`
- `MorningNight` binary flag

---

## 🤖 Models Used

### 1. **Linear Regression**
- Used as a baseline
- Performed poorly without feature scaling/outlier handling
- **RMSE ~ 96.17** before data cleaning

### 2. **Ridge Regression (L2 Regularization)**
- GridSearchCV used to tune `alpha = [0.01, 0.1, 1, 10, 100]`
- Best `alpha = 100`
- **Best RMSE (CV): 9.5941**

### 3. **Lasso Regression (L1 Regularization)**
- Used to perform implicit feature selection
- Best `alpha = 0.01`
- **Best RMSE (CV): 9.5942**

### 4. **XGBoost Regressor**
- Captured non-linear relationships
- Used default hyperparameters initially
- Trained on 70% of cleaned data
- **Final RMSE: 5.38**
- **R² Score: 0.83**, **MAE: $2.37**

---

## 📈 Results Summary

| Model            | R² Score | RMSE   | MAE   | Notes                         |
|------------------|----------|--------|-------|-------------------------------|
| LinearRegression | ~0.01    | ~96    | ~17   | Baseline, high error          |
| Ridge Regression | ~0.82    | 9.59   | -     | L2 regularized model          |
| Lasso Regression | ~0.82    | 9.59   | -     | L1 regularized model          |
| **XGBoost**      | **0.83** | **5.38** | **2.37** | Best-performing model         |

---

## 📁 Directory Structure
