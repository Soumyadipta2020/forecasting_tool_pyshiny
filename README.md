# AI Forecasting Workbench

[![CI](https://github.com/Soumyadipta2020/forecasting_tool_pyshiny/actions/workflows/ci.yml/badge.svg)](https://github.com/Soumyadipta2020/forecasting_tool_pyshiny/actions/workflows/ci.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/Soumyadipta2020/forecasting_tool_pyshiny?style=social)
![GitHub forks](https://img.shields.io/github/forks/Soumyadipta2020/forecasting_tool_pyshiny?style=social)
![GitHub license](https://img.shields.io/github/license/Soumyadipta2020/forecasting_tool_pyshiny)

A PyShiny forecasting application for data quality review, model comparison, validation, explainability, and report export. It supports both time-series forecasting and tabular regression/classification-style workflows from a browser UI.

## Demo

Run the app locally and load the bundled `app/timeseries_demo.csv` dataset to explore the full workflow. Captured screenshots and a short demo script live in [docs/DEMO.md](docs/DEMO.md).

## Highlights

- Upload CSV data or use the bundled sample dataset
- Guided workflow across data quality, summary statistics, forecasting, validation, explainability, and reporting
- Time-series models including Naive, Seasonal Naive, Moving Average, Drift, ARIMA, Auto ARIMA, SARIMA, ETS, Prophet, Theta, Croston, GRNN, Neural Network, AutoML-style regressors, State Space ARIMA, ARCH/GARCH, ARFIMA approximation, and Ensemble
- Tabular models including Linear Regression, GLM, Logistic Regression, LASSO, Elastic Net, Ridge Regression, Random Forest, Gradient Boosting, and SVM
- Holdout and rolling validation, residual diagnostics, anomaly detection, prediction interval quality, and downloadable results
- Optional compact hyperparameter tuning for supported models
- CI coverage for syntax checks and shared modeling helpers

## Tech Stack

- Python 3.10+
- PyShiny and shinywidgets for the web app
- pandas and NumPy for data handling
- Plotly for interactive charts
- statsmodels, Prophet, scikit-learn, SciPy, and arch for forecasting and modeling
- pytest for tests
- GitHub Actions for CI

## Project Structure

```text
.
|-- app/
|   |-- app.py                    # PyShiny app object
|   |-- run.py                    # Local Python runner
|   |-- server.py                 # Main server logic
|   |-- timeseries_demo.csv       # Bundled sample data
|   |-- server_scripts/helpers/   # Shared modeling and helper functions
|   |-- ui_scripts/components/    # UI tab components
|   `-- www/                      # Static assets and custom CSS/JS
|-- docs/                         # Demo and screenshot guidance
|-- tests/                        # Test coverage for shared helpers
|-- pyproject.toml                # Project metadata and dependencies
`-- README.md
```

## Quick Start

From the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m shiny run app/app.py --reload --port 8000
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m shiny run app/app.py --reload --port 8000
```

Then open `http://localhost:8000`.

Alternative dependency install:

```bash
python -m pip install -r requirements.txt
```

## Usage

1. Open the app and choose **Load sample data** or upload a CSV.
2. Review data quality checks for missing values, duplicates, outliers, mixed types, and date gaps.
3. Select a response variable and predictors or a time variable.
4. Choose models manually or apply the recommended settings.
5. Generate forecasts, compare metrics, inspect diagnostics, and export results.

## Data Format

Use CSV files with:

- One response column to forecast or predict
- Optional time/index column for time-series workflows
- Numeric or categorical predictor columns for tabular workflows
- Clean column names without duplicate headers

The bundled sample file is [app/timeseries_demo.csv](app/timeseries_demo.csv).

## Quality Checks

```bash
python -m compileall -q app tests
python -m pytest
```

The GitHub Actions workflow runs dependency installation, syntax checks, and tests on pushes and pull requests.

## Portfolio Notes

This repository is a good candidate to pin on a GitHub profile because it shows:

- A usable deployed-style application, not only scripts
- Clear setup and project structure
- Multiple modeling approaches with validation and reporting
- Tests and CI
- A demo workflow that can be turned into screenshots or a short video

## License

This project is licensed under the GPL-3.0 license. See [LICENSE](LICENSE).
