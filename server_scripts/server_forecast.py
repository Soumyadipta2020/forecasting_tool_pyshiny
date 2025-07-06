"""
Forecasting model logic for the AI Forecasting Application.
Contains Prophet and ARIMA forecast functions.
"""
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from server_scripts.global_helpers import calculate_metrics

def run_prophet_forecast(ts_data, time_var, target_var, horizon, output):
    prophet_df = ts_data.rename(columns={time_var: 'ds', target_var: 'y'})
    if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)
    @output
    @output.plot
    def forecast_plot():
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        ax.plot(prophet_df['ds'], prophet_df['y'], 'b-', linewidth=2, alpha=0.8, label='Actual')
        ax.plot(forecast['ds'].tail(horizon), forecast['yhat'].tail(horizon), 'r-', linewidth=2, alpha=0.8, label='Forecast')
        ax.fill_between(
            forecast['ds'].tail(horizon+10),
            forecast['yhat_lower'].tail(horizon+10),
            forecast['yhat_upper'].tail(horizon+10),
            color='red', alpha=0.2, label='95% Confidence Interval')
        ax.set_title(f'Prophet Forecast for {target_var} (Next {horizon} periods)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(target_var, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    actual = prophet_df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    metrics_df = calculate_metrics(actual, predicted)
    @output
    @output.table
    def forecast_metrics():
        return metrics_df

def run_arima_forecast(ts_data, time_var, target_var, horizon, output):
    ts_values = ts_data[target_var].values
    model = ARIMA(ts_values, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=horizon)
    @output
    @output.plot
    def forecast_plot():
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        if pd.api.types.is_datetime64_any_dtype(ts_data[time_var]):
            x_actual = ts_data[time_var]
            last_date = x_actual.iloc[-1]
            date_range = pd.date_range(start=last_date, periods=horizon+1)[1:]
            x_forecast = date_range
            ax.plot(x_actual, ts_values, 'b-', linewidth=2, alpha=0.8, label='Actual')
            ax.plot(x_forecast, forecast_values, 'r-', linewidth=2, alpha=0.8, label='Forecast')
        else:
            ax.plot(range(len(ts_values)), ts_values, 'b-', linewidth=2, alpha=0.8, label='Actual')
            ax.plot(range(len(ts_values), len(ts_values) + horizon), forecast_values, 'r-', linewidth=2, alpha=0.8, label='Forecast')
        ax.set_title(f'ARIMA Forecast for {target_var} (Next {horizon} periods)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(target_var, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    predicted = model_fit.fittedvalues
    actual = ts_values[1:len(predicted)+1]
    metrics_df = calculate_metrics(actual, predicted)
    @output
    @output.table
    def forecast_metrics():
        return metrics_df
