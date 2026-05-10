# Crime Forecasting Project

This project predicts crime trends using historical crime data (2020–2025).  
It combines **machine learning forecasting** with **Power BI dashboards** for visualization.

\---

## Problem Statement

City officials and law enforcement agencies need data-driven insights to anticipate crime trends and allocate resources effectively.

\---

## Approach

1. **Data Preparation**

   * Cleaned and transformed crime records (2020–2025).
   * Split datasets into training and test sets.
2. **Model Training**

   * Machine learning model trained on data (2020–2022).
   * Tested on 2023 for validation (Jan–Dec 2023).
3. **Visualization**

   * Built interactive Power BI dashboards to show trends and forecasts.
   * Forecast plot generated in Python showing actual vs predicted (Jan–Dec 2023).

\---

## Tools \& Technologies

* Python (Pandas, Scikit-learn, Matplotlib, XGBoost)
* Power BI for visualization
* Jupyter Notebook for experiments

\---

## Results

* MAE (Jan–Dec 2023): **384.39**
* RMSE (Jan–Dec 2023): **437.15**

**Forecast Plot (actuals up to Dec 2023):**  
!\[Forecast Plot](./reports/forecast\_plot\_v1\_20250829.png)

**Download Forecast CSV:** [forecast\_2023.csv](./reports/forecast_2023.csv)  
**Download Forecast Plot:** [forecast\_plot\_v1\_20250829.png](./reports/forecast_plot_v1_20250829.png)

Evaluation metrics show that the model predicts monthly crime counts with an average error of \~384 cases per month and a root mean squared error of \~437 cases.

\---

## Dashboard

* Exported Power BI report as PDF: [Crime Dashboard PDF](./reports/crime_forecast.pdf)
* Screenshots of interactive dashboard:



!\[Crime Dashboard - Page 1](./reports/crime\_dashboard\_page1.png)



!\[Crime Dashboard - Page 2](./reports/crime\_dashboard\_page2.png)



!\[Crime Dashboard - Page 3](./reports/crime\_dashboard\_page3.png)



\---

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/Christo1011/data-analytics-portfolio.git
cd crime-forecasting
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training pipeline or notebook:

```bash
python pipeline.py
# or
jupyter notebook notebooks/crime\\\\\\\\\\\\\\\_forecast\\\\\\\\\\\\\\\_2023.ipynb

\\\\\\\\---

## Next Steps

\\\\\\\* Add GitHub Actions workflow for automation
\\\\\\\* Deploy FastAPI service to allow live predictions
\\\\\\\* Expand to real-time streaming data

\\\\\\\\---

Project Structure
crime-forecasting/
│
├── data/                # Raw and processed datasets
│   └── processed/
│       └── cleaned\\\\\\\\\\\\\\\_data.csv
├── notebooks/           # Jupyter notebooks (training \\\\\\\\\\\\\\\& testing)
│   ├── crime\\\\\\\\\\\\\\\_forecast\\\\\\\\\\\\\\\_2023.ipynb
│   ├── data\\\\\\\\\\\\\\\_cleaning.ipynb
│   ├── FastApi.ipynb
│   ├── pipeline.ipynb
│   └── xgb\\\\\\\\\\\\\\\_crime\\\\\\\\\\\\\\\_model.ipynb
├── reports/             # Power BI exports \\\\\\\\\\\\\\\& screenshots
│   ├── crime\\\\\\\\\\\\\\\_forecast.pdf
│   ├── crime\\\\\\\\\\\\\\\_forecast.pbix
│   ├── forecast\\\\\\\\\\\\\\\_2023.csv
│   ├── forecast\\\\\\\\\\\\\\\_plot\\\\\\\\\\\\\\\_v1\\\\\\\\\\\\\\\_20250829.png
│   ├── crime\\\\\\\\\\\\\\\_dashboard\\\\\\\\\\\\\\\_page1.png
│   ├── crime\\\\\\\\\\\\\\\_dashboard\\\\\\\\\\\\\\\_page2.png
│   └── crime\\\\\\\\\\\\\\\_dashboard\\\\\\\\\\\\\\\_page3.png
├── scripts/             # Python scripts (pipeline, model, FastAPI)
│   ├── crime\\\\\\\\\\\\\\\_forecast\\\\\\\\\\\\\\\_2023.py
│   ├── data\\\\\\\\\\\\\\\_cleaning.py
│   ├── FastApi.py
│   ├── xgb\\\\\\\\\\\\\\\_crime\\\\\\\\\\\\\\\_model.py
│   └── xgb\\\\\\\\\\\\\\\_crime\\\\\\\\\\\\\\\_model\\\\\\\\\\\\\\\_v1\\\\\\\\\\\\\\\_20250829.pkl
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Author

Christodoulos Nicolaou  
Data Analyst | Aspiring Data Engineer

