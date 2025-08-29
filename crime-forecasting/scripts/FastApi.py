#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

"""
FastAPI App for Crime Forecast Dashboard
----------------------------------------
This version:
- Loads cleaned crime data
- Loads pre-trained XGBoost model from pipeline
- Generates predictions for each area without retraining
- Provides interactive Plotly visualization and JSON API
"""

# ==========================
# Imports
# ==========================
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import pandas as pd
import plotly.graph_objects as go
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import textwrap

# ==========================
# Config & Paths
# ==========================
BASE_DIR = Path(os.getcwd()).parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = BASE_DIR / "scripts" / "xgb_crime_model.pkl"

# Load cleaned data
historical_df = pd.read_csv(DATA_PATH)
historical_df["date_occ"] = pd.to_datetime(historical_df["date_occ"])
historical_df.sort_values("date_occ", inplace=True)

# Load trained model
model = joblib.load(MODEL_PATH)

# Exclude unreliable 2024 data
EXCLUDE_TEXT = (
    "The 2024 crime data is excluded from training and testing because it is known to be unreliable. "
    "Starting March 7, 2024, the LAPD switched to a new Records Management System (NIBRS mandate). "
    "During this transition, some incidents were missing or delayed in reporting. "
    "To avoid skewing the model, only reliable 2020–2023 data is used for training."
)
df_train = historical_df[historical_df["date_occ"].dt.year != 2024].copy()
df_train["year"] = df_train["date_occ"].dt.year
df_train["month"] = df_train["date_occ"].dt.month

# List of areas for dropdown menu
areas = sorted(df_train["area_name"].unique())
areas.insert(0, "All")  # Add "All" option

# ==========================
# Helper function
# ==========================
def prepare_monthly_data(df, area_name=None):
    """
    Aggregate monthly crime counts for a specific area (or all areas).
    """
    if area_name and area_name != "All":
        df = df[df["area_name"] == area_name]
    monthly = df.groupby(["year", "month"]).size().reset_index(name="y")
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    return monthly


def generate_predictions(monthly_df):
    """
    Generate predictions using pre-trained model.
    """
    feature_cols = ['year', 'month']  # Only year and month for X
    X = monthly_df[feature_cols]
    preds = model.predict(X)
    return preds

# ==========================
# FastAPI App
# ==========================
app = FastAPI(title="Crime Forecast Dashboard", version="1.0")

@app.get("/predict_plot", response_class=HTMLResponse)
def predict_plot(year: Optional[int] = Query(None), month: Optional[int] = Query(None)):
    fig = go.Figure()
    all_traces = []

    for area_name in areas:
        monthly_df_area = prepare_monthly_data(df_train, area_name)

        # Skip if insufficient data
        if len(monthly_df_area) <= 1:
            continue

        # Generate predictions
        preds = generate_predictions(monthly_df_area)

        # Actual trace
        actual_trace = go.Scatter(
            x=monthly_df_area["date"],
            y=monthly_df_area["y"],
            mode="lines+markers",
            name=f"Actual ({area_name})",
            line=dict(color="blue"),
            visible=True if area_name == "All" else False,
        )
        all_traces.append(actual_trace)

        # Predicted trace
        pred_trace = go.Scatter(
            x=monthly_df_area["date"],
            y=preds,
            mode="lines+markers",
            name=f"Predicted ({area_name})",
            line=dict(color="orange", dash="dash"),
            visible=True if area_name == "All" else False,
        )
        all_traces.append(pred_trace)

    # Add all traces to figure
    for trace in all_traces:
        fig.add_trace(trace)

    # Dropdown menu
    buttons = []
    for i, area_name in enumerate(areas):
        vis = [False] * len(all_traces)
        idx = i * 2
        if idx < len(all_traces): vis[idx] = True
        if idx + 1 < len(all_traces): vis[idx + 1] = True

        graph_title = "Actual and Predicted Data" if area_name == "All" else f"Crime Data for {area_name}"
        buttons.append(dict(
            label=area_name,
            method="update",
            args=[{"visible": vis}, {"title": {"text": graph_title, "x": 0.5, "xanchor": "center"}}]
        ))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.85, y=1.15)],
        title={"text": "Actual and Predicted Data", "x": 0.5, "xanchor": "center"},
        xaxis_title="Date",
        yaxis_title="Crime Count",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    wrapped_text = "<br>".join(textwrap.wrap(EXCLUDE_TEXT, width=120))
    html_content = f"""
    <html>
        <head><title>Crime Forecast Dashboard</title></head>
        <body>
            <div style='margin-bottom:20px; font-size:14px;'>{wrapped_text}</div>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
    </html>
    """
    return HTMLResponse(html_content)


@app.get("/api/predictions", response_class=JSONResponse)
def api_predictions():
    results = []
    for area_name in df_train["area_name"].unique():
        monthly_df_area = prepare_monthly_data(df_train, area_name)
        if len(monthly_df_area) <= 1:
            continue

        preds = generate_predictions(monthly_df_area)

        # Append actual values
        for _, row in monthly_df_area.iterrows():
            results.append({
                "area_name": area_name,
                "date": str(row["date"].date()),
                "actual": int(row["y"]),
                "predicted": None
            })
        # Append forecasted values
        for i, row in monthly_df_area.iterrows():
            results.append({
                "area_name": area_name,
                "date": str(row["date"].date()),
                "actual": None,
                "predicted": float(preds[i])
            })

    return {"data": results}


# ==========================
# Run server
# ==========================
if __name__ == "__main__":
    host = "127.0.0.2"
    port = 8000

    # Auto-open browser
    url = f"http://{host}:{port}/predict_plot"
    def open_browser():
        time.sleep(2)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host=host, port=port)


# In[ ]:




