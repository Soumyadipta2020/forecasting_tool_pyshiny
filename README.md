# AI Forecasting App

## Directory Structure Update (July 2025)

The `helpers` folder and `global_helpers.py` have been moved into a new `server_scripts` folder for better modularity. Please update your imports accordingly:

- Use `from server_scripts.global_helpers import ...` instead of `from global_helpers import ...`
- Use `from server_scripts.helpers import ...` for helper modules.

A PyShiny application for time series forecasting using various AI and statistical models.

## Features

-   Upload CSV data or use the bundled sample time-series dataset
-   Guided workflow from data quality review to summary statistics, forecasting, and report export
-   Data quality checks for missing values, duplicates, outliers, mixed numeric/text columns, and date gaps
-   Interactive visualizations and summary statistics
-   Time-series forecasting with baseline, statistical, machine-learning-style, Prophet, and ensemble model options
-   Non-time-series regression/classification-style model comparison
-   Holdout and rolling validation, residual diagnostics, anomaly detection, explainability, and forecast metrics
-   Downloadable forecast results, summary statistics, and full HTML reports

## Installation

1.  Clone this repository
2.  Install the required dependencies:

``` bash
pip install -r requirements.txt
```

3.  No database is required for the bundled sample dataset.

## Usage

Run the application:

``` bash
shiny run app.py
```

Then open your browser and navigate to http://localhost:8000

## Data Format

The application expects CSV files with at least one time column and one or more numeric columns for forecasting. You can download a template from the application.

## Models

The forecasting workspace includes options such as Naive, Seasonal Naive, Moving Average, Drift, ARIMA, SARIMA, ETS, Prophet, GRNN, ARFIMA, ARCH/GARCH, Neural Network, AutoML-style regressors, State Space ARIMA, and Ensemble. For tabular data, the app includes Linear Regression, GLM, Logistic Regression, LASSO, and Ridge Regression.