# AI Forecasting App

A PyShiny application for time series forecasting using various AI and statistical models.

## Features

- Upload CSV data or use sample data from MongoDB
- Interactive data visualization and editing
- Multiple forecasting models:
  - Prophet
  - Auto ARIMA
  - LSTM (Deep Learning)
  - AutoML (H2O)
  - ARFIMA
- Forecast metrics and evaluation
- Modern, responsive UI

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have MongoDB installed and running (optional, for sample data functionality)

## Usage

Run the application:

```bash
shiny run app.py
```

Then open your browser and navigate to http://localhost:8000

## Data Format

The application expects CSV files with at least one time column and one or more numeric columns for forecasting. You can download a template from the application.

## Models

### Prophet
Facebook's Prophet model for time series forecasting with support for yearly, weekly, and daily seasonality.

### Auto ARIMA
Automatic ARIMA model selection with support for seasonal components.

### LSTM
Long Short-Term Memory neural network for sequence prediction.

### AutoML
H2O AutoML for automated machine learning model selection and training.

### ARFIMA
AutoRegressive Fractionally Integrated Moving Average model.

## Author

Soumyadipta Das

## Version

0.03.3