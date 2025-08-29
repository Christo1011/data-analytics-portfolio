#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

"""
Pipeline to train and save crime forecast model
------------------------------------------------
This script:
- Installs required packages if missing
- Loads cleaned crime data
- Engineers features
- Trains XGBoost model
- Saves the trained model, 2023 forecast, and forecast plot
"""

# ==========================
# Step 0: Install required packages
# ==========================
import subprocess
import sys
import importlib
import os

required_packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "matplotlib"
]

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ==========================
# Step 1: Imports
# ==========================
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add scripts folder to sys.path so we can import functions
BASE_DIR = Path(os.getcwd())
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from crime_forecast_2023 import load_cleaned_data, engineer_features, train_evaluate_model

# ==========================
# Step 2: Paths & Versioning
# ==========================
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
FORECAST_PATH = BASE_DIR / "reports" / "forecast_2023.csv"

MODEL_VERSION = "v1"
today_str = datetime.today().strftime("%Y%m%d")
MODEL_PATH = BASE_DIR / "scripts" / f"xgb_crime_model_{MODEL_VERSION}_{today_str}.pkl"
PLOT_PATH = BASE_DIR / "reports" / f"forecast_plot_{MODEL_VERSION}_{today_str}.png"

# ==========================
# Step 3: Train Pipeline Function
# ==========================
def train_forecast_pipeline():
    # Load data
    df = load_cleaned_data(DATA_PATH)

    # Feature engineering
    monthly_counts = engineer_features(df)

    # Train and evaluate
    train_end = "2022-12-31"
    test_start = "2023-01-01"
    test_end = "2023-12-31"
    model, metrics, pred_df = train_evaluate_model(
        monthly_counts=monthly_counts,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        save_path=MODEL_PATH
    )

    # Save 2023 forecast
    pred_df.to_csv(FORECAST_PATH, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"2023 forecast saved to: {FORECAST_PATH}")
    print(f"Metrics: {metrics}")

    # ==========================
    # Step 4: Save Forecast Plot (Actuals up to Dec 31, 2023)
    # ==========================
    mask_actuals = monthly_counts['date_occ'] <= pd.Timestamp('2023-12-31')

    plt.figure(figsize=(12, 6))
    # Actuals line (2020 → Dec 2023)
    plt.plot(
        monthly_counts.loc[mask_actuals, 'date_occ'],
        monthly_counts.loc[mask_actuals, 'count'],
        label='Actual (2020–Dec 2023)',
        marker='o'
    )

    # Predictions line (2023 → Dec 2025)
    plt.plot(
        pred_df['date_occ'],
        pred_df['Predicted'],
        label='Predicted (2023–Dec 2025)',
        marker='x',
        linestyle='--'
    )

    plt.title('Crime Forecast - Total (2020–2025, actuals up to Dec 2023)')
    plt.xlabel('Date')
    plt.ylabel('Crime Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Forecast plot saved to: {PLOT_PATH}")

    return model, pred_df, metrics

# ==========================
# Step 5: Run pipeline
# ==========================
if __name__ == "__main__":
    train_forecast_pipeline()


# In[ ]:




