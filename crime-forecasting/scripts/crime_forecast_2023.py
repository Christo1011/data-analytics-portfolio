#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
crime_forecast_2023.py

This script contains reusable functions to:
1. Load and preprocess cleaned crime data.
2. Engineer features for monthly crime prediction.
3. Train and evaluate an XGBoost model.
4. Save the trained model.

The 2024 data is excluded due to known reporting issues.
"""

# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Load & preprocess data
# =========================
def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned crime data CSV and preprocess date columns.

    Parameters:
        file_path (str): Path to the cleaned CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'date_occ' column as datetime.
    """
    df = pd.read_csv(file_path, parse_dates=['date_occ'])
    df.columns = df.columns.str.strip()  # remove whitespace
    df = df.dropna(subset=['date_occ'])  # drop rows with missing dates
    return df

# =========================
# Feature engineering
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly crime counts and add temporal features.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        pd.DataFrame: Monthly aggregated DataFrame with features.
    """
    # Aggregate by month (month-end)
    monthly_counts = df.groupby(pd.Grouper(key='date_occ', freq='M')).size().reset_index(name='count')

    # Year and month columns
    monthly_counts['year'] = monthly_counts['date_occ'].dt.year
    monthly_counts['month'] = monthly_counts['date_occ'].dt.month

    # Seasonal features using sin/cos
    monthly_counts['month_sin'] = np.sin(2 * np.pi * monthly_counts['month'] / 12)
    monthly_counts['month_cos'] = np.cos(2 * np.pi * monthly_counts['month'] / 12)

    # Lag features (1–6 months)
    for lag in range(1, 7):
        monthly_counts[f'lag_{lag}'] = monthly_counts['count'].shift(lag).fillna(0)

    # 3-month rolling average
    monthly_counts['rolling_3'] = monthly_counts['count'].rolling(3).mean().fillna(0)

    return monthly_counts

# =========================
# Train and evaluate model
# =========================
def train_evaluate_model(monthly_counts: pd.DataFrame, train_end: str, test_start: str, test_end: str, save_path: str):
    """
    Train XGBoost model on monthly crime data and evaluate on a test period.

    Parameters:
        monthly_counts (pd.DataFrame): DataFrame with engineered features.
        train_end (str): Last date for training (YYYY-MM-DD).
        test_start (str): First date for test period (YYYY-MM-DD).
        test_end (str): Last date for test period (YYYY-MM-DD).
        save_path (str): Path to save the trained XGBoost model.

    Returns:
        tuple: Trained model, test metrics dict, DataFrame with predictions.
    """
    # Training set
    train_df = monthly_counts[monthly_counts['date_occ'] <= pd.Timestamp(train_end)]

    # Test set
    test_df = monthly_counts[(monthly_counts['date_occ'] >= pd.Timestamp(test_start)) &
                             (monthly_counts['date_occ'] <= pd.Timestamp(test_end))]

    # Features
    feature_cols = ['year', 'month', 'month_sin', 'month_cos'] + [f'lag_{i}' for i in range(1, 7)] + ['rolling_3']
    X_train = train_df[feature_cols]
    y_train = train_df['count']
    X_test = test_df[feature_cols]
    y_test = test_df['count']

    # Initialize and train XGBoost
    model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {"MAE": mae, "RMSE": rmse}

    # Save model
    joblib.dump(model, save_path)

    # Prepare predictions DataFrame
    pred_df = test_df.copy()
    pred_df['Predicted'] = y_pred
    pred_df['Year'] = pred_df['date_occ'].dt.year
    pred_df['Month'] = pred_df['date_occ'].dt.month
    pred_df = pred_df[['Year', 'Month', 'date_occ', 'count', 'Predicted']].rename(columns={'count': 'Actual'})

    return model, metrics, pred_df

# =========================
# Plot actual vs predicted
# =========================
def plot_predictions(pred_df: pd.DataFrame, title: str = "Crime Forecast - Jul–Dec 2023"):
    """
    Plot actual vs predicted monthly crime counts.

    Parameters:
        pred_df (pd.DataFrame): DataFrame with 'date_occ', 'Actual', 'Predicted' columns.
        title (str): Plot title.
    """
    plt.figure(figsize=(12,6))
    plt.plot(pred_df['date_occ'], pred_df['Actual'], label='Actual', marker='o')
    plt.plot(pred_df['date_occ'], pred_df['Predicted'], label='Predicted', marker='x', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Crime Count')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_forecast_model(data_path: str, model_save_path: str, forecast_save_path: str,
                         train_end: str = "2023-06-30",
                         test_start: str = "2023-07-01",
                         test_end: str = "2023-12-31"):
    """
    Complete workflow to train XGBoost crime forecast model and save results.

    Steps:
    1. Load cleaned crime data.
    2. Engineer monthly features.
    3. Train XGBoost on data up to `train_end`.
    4. Predict crime counts for test period (`test_start` → `test_end`).
    5. Save trained model and test predictions (forecast CSV).

    Parameters:
        data_path (str): Path to cleaned CSV data.
        model_save_path (str): Path to save trained XGBoost model (.pkl).
        forecast_save_path (str): Path to save test predictions CSV.
        train_end (str): Last date for training (YYYY-MM-DD).
        test_start (str): First date of test/forecast period (YYYY-MM-DD).
        test_end (str): Last date of test/forecast period (YYYY-MM-DD).

    Returns:
        tuple: trained model, metrics dictionary, predictions DataFrame
    """
    # Load cleaned data
    df = load_cleaned_data(data_path)

    # Engineer features
    monthly_counts = engineer_features(df)

    # Train and evaluate
    model, metrics, pred_df = train_evaluate_model(
        monthly_counts,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        save_path=model_save_path
    )

    print(f"Model trained and saved to: {model_save_path}")
    print(f"Test metrics: {metrics}")

    # Save forecast CSV
    pred_df.to_csv(forecast_save_path, index=False)
    print(f"Forecast CSV saved to: {forecast_save_path}")

    # Optional: plot predictions
    plot_predictions(pred_df, title=f"Crime Forecast ({test_start} → {test_end})")

    return model, metrics, pred_df


