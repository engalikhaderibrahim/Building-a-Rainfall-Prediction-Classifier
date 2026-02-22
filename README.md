# 🌦️ Australian Rainfall Prediction: End-to-End ML Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kyYzQDARFowJGMbOR9JE8A27-aJn0MlK?usp=sharing)

## 📖 Project Overview
This repository features a robust, end-to-end Machine Learning pipeline designed to predict whether it will rain tomorrow in Australia. Leveraging a decade of daily weather observations, the project demonstrates advanced data science methodologies, including smart data imputation, feature engineering, pipeline construction, and handling imbalanced datasets.

## 🎯 Objectives
- **Predictive Modeling:** Build a classification model to accurately predict daily rainfall (Binary Classification: `Yes` or `No`).
- **Pipeline Integration:** Utilize `scikit-learn` pipelines to ensure reproducible and leak-free data transformations.
- **Imbalanced Data Handling:** Address the natural skew in weather data (more sunny days than rainy days) using class-weight balancing and appropriate evaluation metrics like ROC-AUC.

## 🛠️ Technologies & Libraries
- **Language:** Python 3
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (Random Forest, Logistic Regression, GridSearchCV, Pipelines)
- **Data Visualization:** `matplotlib`, `seaborn`

## 🧠 Methodology
1. **Data Preprocessing & Imputation:**
   - Deployed `SimpleImputer` to salvage valuable data points instead of discarding missing rows.
   - Used **Median** imputation for numerical features and **Most Frequent** for categorical variables.
   - Applied `StandardScaler` for numerical scaling and `OneHotEncoder` for categorical encoding within a seamless `ColumnTransformer`.

2. **Feature Engineering:**
   - **Temporal Features:** Extracted `Season` from the `Date` column to capture cyclical weather patterns.
   - **Differential Metrics:** Engineered highly predictive features based on domain knowledge, such as:
     - `TempDiff` (Max vs. Min Temperature)
     - `PressureDiff` (Afternoon vs. Morning Atmospheric Pressure)
     - `HumidityDiff` (Afternoon vs. Morning Humidity)

3. **Model Selection & Hyperparameter Tuning:**
   - Trained and compared **Random Forest** and **Logistic Regression** classifiers.
   - Used `StratifiedKFold` cross-validation to maintain class distribution across splits.
   - Optimized hyperparameters using `GridSearchCV` with a focus on maximizing the **ROC-AUC** score rather than simple accuracy.

## 📊 Key Results
The finalized models yielded highly competitive results:
- **Cross-Validation ROC-AUC:** `~0.89`
- **Test Set Accuracy:** `~83% - 85%`
- **Insights:** The models showed a strong ability to distinguish between classes. Feature importance analysis highlighted `Humidity3pm`, `PressureDiff`, and `WindGustSpeed` as the strongest predictors of imminent rainfall.

## 🚀 How to Run
1. **Run in Colab:** Click the "Open in Colab" badge at the top of this page to view and interact with the notebook directly.
