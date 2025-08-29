#!/usr/bin/env python
# coding: utf-8

# In[15]:


"""
data_cleaning.py
----------------
Module for fetching and cleaning Los Angeles crime data from the public API.

Steps:
1. Fetch raw data from LA city API in batches.
2. Clean and standardize string columns.
3. Drop columns with too many missing values (> 80%).
4. Normalize victim demographics (sex, descent).
5. Clean and validate date and time fields.
6. Save the cleaned dataset to CSV.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests


# =========================
# API Settings
# =========================
BASE_URL = "https://data.lacity.org/resource/2nrs-mtv8.json"
LIMIT = 50000  # max rows per API call


def fetch_data():
    """
    Fetches all rows from the LA crime API in batches.
    
    Returns:
        pd.DataFrame: Raw DataFrame of all crime records.
    """
    offset = 0
    all_data = []
    print("Fetching Data...")

    while True:
        url = f"{BASE_URL}?$limit={LIMIT}&$offset={offset}"
        response = requests.get(url)
        response.raise_for_status()

        batch = response.json()
        if not batch:  # Stop when no more rows
            break

        all_data.extend(batch)
        offset += LIMIT
        print(f"Fetched {len(batch)} rows, total: {len(all_data)}")

    df = pd.DataFrame(all_data)
    return df


def clean_spaces(val):
    """Removes extra spaces from string values."""
    if isinstance(val, str):
        return " ".join(val.split())
    return val


def nan_percentage_and_drop(df, column_name, threshold=80):
    """
    Drops a column if missing values exceed the threshold.
    
    Args:
        df (pd.DataFrame): DataFrame to process.
        column_name (str): Column name.
        threshold (float): Percentage threshold for dropping.
    
    Returns:
        pd.DataFrame: Modified DataFrame.
    """
    if column_name not in df.columns:
        return df

    total_rows = len(df)
    missing_count = df[column_name].isna().sum()
    percent_missing = (missing_count / total_rows) * 100

    if percent_missing > threshold:
        df = df.drop(columns=[column_name])
        print(f"Dropped column '{column_name}' (>{threshold}% missing).")

    return df


def clean_data(df):
    """
    Cleans and preprocesses the raw crime data.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Strip whitespace in string columns
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(include="object").apply(
        lambda col: col.map(clean_spaces)
    )

    # Drop columns with excessive NaN
    temp_df = df.copy()
    for col in list(temp_df.columns):
        temp_df = nan_percentage_and_drop(temp_df, col)
    df = temp_df

    # Age cleanup
    df["vict_age"] = pd.to_numeric(df["vict_age"], errors="coerce")
    df["vict_age"] = df["vict_age"].apply(
        lambda x: np.nan if pd.notna(x) and (x < 10 or x > 100) else x
    )

    # Map victim demographics
    descent_dict = {
        "A": "Other Asian", "B": "Black", "C": "Chinese", "D": "Cambodian",
        "F": "Filipino", "G": "Guamanian", "H": "Hispanic/Latin/Mexican",
        "I": "American Indian/Alaskan Native", "J": "Japanese", "K": "Korean",
        "L": "Laotian", "O": "Other", "P": "Pacific Islander", "S": "Samoan",
        "U": "Hawaiian", "V": "Vietnamese", "W": "White", "X": "Unknown", "Z": "Asian Indian"
    }
    sex_dict = {"F": "Female", "M": "Male", "X": "Unknown"}

    df["vict_descent"] = df["vict_descent"].map(descent_dict).fillna("Unknown")
    df["vict_sex"] = df["vict_sex"].map(sex_dict).fillna("Unknown")

    # Date processing
    df = df.copy()
    df["DATE OCC parsed"] = pd.to_datetime(df["date_occ"], errors="coerce")
    df = df[~df["DATE OCC parsed"].isna()]
    df["date_occ"] = df["DATE OCC parsed"].dt.date

    df["Date Rptd parsed"] = pd.to_datetime(df["date_rptd"], errors="coerce")
    df = df[~df["Date Rptd parsed"].isna()]
    df["date_rptd"] = df["Date Rptd parsed"].dt.date

    # Drop 2025 data
    mask_2025 = (
        (pd.to_datetime(df["date_occ"]).dt.year == 2025)
        | (pd.to_datetime(df["date_rptd"]).dt.year == 2025)
    )
    df = df[~mask_2025]

    # Time cleanup
    df["TIME OCC num"] = pd.to_numeric(df["time_occ"], errors="coerce")
    df = df[df["TIME OCC num"] > 99]
    df["time_occ"] = df["TIME OCC num"].astype(int).astype(str).str.zfill(4)
    df["time_occ"] = df["time_occ"].str[:2] + ":" + df["time_occ"].str[2:]

    # Drop helper columns
    df.drop(columns=["DATE OCC parsed", "Date Rptd parsed", "TIME OCC num"], inplace=True)

    return df


def save_data(df, filename="cleaned_data.csv"):
    """
    Saves cleaned DataFrame to the processed data folder.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        filename (str): Output file name.
    """
    processed_folder = Path.cwd().parent / "data" / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    file_path = processed_folder / filename
    df.to_csv(file_path, index=False)
    print(f"DataFrame exported to: {file_path}")

def clean_and_save_data(output_path: Path):
    """Fetches raw data, cleans it, and saves to output_path"""
    raw_df = fetch_data()
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df, output_path)
    
if __name__ == "__main__":
    # Run the full pipeline when executed directly
    raw_df = fetch_data()
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df)


# In[ ]:




