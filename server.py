#!/usr/bin/env python3
"""
Server-side logic for the AI Forecasting Application.

This module defines the server functions that handle reactive data flow,
data processing, and forecasting models.
"""

# ===== IMPORTS =====
from shiny import reactive, render
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from global_helpers import calculate_metrics

# ===== MAIN SERVER FUNCTION =====
def server_function(input, output, session):
    """
    Main server function for the Shiny application.
    
    Handles reactive data flow, user inputs, and output rendering.
    
    Parameters:
    -----------
    input : InputInterface
        User input values from UI components
    output : OutputInterface 
        Interface for rendering outputs
    session : SessionContext
        The current session context
    """
    # ----- Reactive Values -----
    # Store uploaded data
    data = reactive.Value(None)
    
    # ----- File Upload Handler -----
    @reactive.effect
    def _():
        """Handle file upload and update variable dropdowns."""
        file_info = input.file()
        if file_info and len(file_info) > 0:
            file_path = file_info[0]["datapath"]
            df = pd.read_csv(file_path)
            data.set(df)
            
            # Update time variable dropdown
            session.ui.update_select(
                "time_variable", 
                choices=df.columns.tolist(),
                selected=df.columns[0] if len(df.columns) > 0 else None
            )
            
            # Update target variable dropdown with numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            session.ui.update_select(
                "target_variable", 
                choices=numeric_cols,
                selected=numeric_cols[0] if len(numeric_cols) > 0 else None
            )
    
    # ----- Data Preview -----
    @output
    @render.data_frame
    def uploaded_data():
        """Render the uploaded data preview."""
        if data.get() is not None:
            return data.get()
        return pd.DataFrame()
    
    # ----- Forecast Runner -----
    @reactive.effect
    def _():
        """
        Process forecast request when button is clicked.
        
        Collects input parameters, validates them, and calls the
        appropriate forecasting model function.
        """
        # Only run when button is clicked
        if not input.run_forecast() or data.get() is None:
            return
        
        df = data.get()
        time_var = input.time_variable()
        target_var = input.target_variable()
        horizon = input.forecast_horizon()
        model_type = input.forecast_model()
        
        # Check if the required variables are selected
        if not time_var or not target_var or time_var not in df.columns or target_var not in df.columns:
            return
        
        # Prepare time series data
        ts_data = df[[time_var, target_var]].copy()
        ts_data = ts_data.sort_values(by=time_var)
        
        # Run the selected forecasting model
        if model_type == "prophet":
            run_prophet_forecast(ts_data, time_var, target_var, horizon, output)
        elif model_type == "auto_arima":
            run_arima_forecast(ts_data, time_var, target_var, horizon, output)
    
    # ----- Initial Plot and Metrics -----
    @output
    @render.plot
    def forecast_plot():
        """Render the initial empty forecast plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5, 0.5,
            "Upload data and click 'Run Forecast' to see results",
            ha='center', va='center',
            transform=ax.transAxes
        )
        return fig
    
    @output
    @render.table
    def forecast_metrics():
        """Render the initial empty metrics table."""
        return pd.DataFrame({
            'Metric': ['Note'],
            'Value': ["Run a forecast to see metrics"]
        })

# ===== FORECASTING MODELS =====
def run_prophet_forecast(ts_data, time_var, target_var, horizon, output):
    """
    Run Facebook Prophet forecasting model.
    
    Parameters:
    -----------
    ts_data : pandas.DataFrame
        Time series data for forecasting
    time_var : str
        Column name for time variable
    target_var : str
        Column name for target variable
    horizon : int
        Number of periods to forecast
    output : OutputInterface
        Interface for rendering outputs
    """
    # ----- Data Preparation -----
    prophet_df = ts_data.rename(columns={time_var: 'ds', target_var: 'y'})
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # ----- Model Training -----    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    
    # ----- Forecast Generation -----
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)
    
    # ----- Plot Rendering -----
    @output
    @render.plot
    def forecast_plot():
        """Render the Prophet forecast plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual values
        ax.plot(prophet_df['ds'], prophet_df['y'], 'b-', label='Actual')
        
        # Plot forecast
        ax.plot(forecast['ds'].tail(horizon), forecast['yhat'].tail(horizon), 'r-', label='Forecast')
        
        # Plot prediction intervals
        ax.fill_between(
            forecast['ds'].tail(horizon+10),
            forecast['yhat_lower'].tail(horizon+10),
            forecast['yhat_upper'].tail(horizon+10),
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        
        ax.set_title(f'Forecast for {target_var} (Next {horizon} periods)')
        ax.set_xlabel('Time')
        ax.set_ylabel(target_var)
        ax.legend()
        
        return fig
    
    # ----- Metrics Calculation -----
    actual = prophet_df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    
    metrics_df = calculate_metrics(actual, predicted)
    
    @output
    @render.table
    def forecast_metrics():
        """Render the Prophet forecast metrics table."""
        return metrics_df

def run_arima_forecast(ts_data, time_var, target_var, horizon, output):
    """
    Run Auto ARIMA forecasting model.
    
    Parameters:
    -----------
    ts_data : pandas.DataFrame
        Time series data for forecasting
    time_var : str
        Column name for time variable
    target_var : str
        Column name for target variable
    horizon : int
        Number of periods to forecast
    output : OutputInterface
        Interface for rendering outputs
    """
    # ----- Data Preparation -----
    ts_values = ts_data[target_var].values
    
    # ----- Model Training -----
    model = ARIMA(ts_values, order=(1, 1, 1))
    model_fit = model.fit()
    
    # ----- Forecast Generation -----
    forecast_values = model_fit.forecast(steps=horizon)
    
    # ----- Plot Rendering -----
    @output
    @render.plot
    def forecast_plot():
        """Render the ARIMA forecast plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual values
        ax.plot(range(len(ts_values)), ts_values, 'b-', label='Actual')
        
        # Plot forecast
        ax.plot(
            range(len(ts_values), len(ts_values) + horizon),
            forecast_values,
            'r-',
            label='Forecast'
        )
        
        ax.set_title(f'Forecast for {target_var} (Next {horizon} periods)')
        ax.set_xlabel('Time')
        ax.set_ylabel(target_var)
        ax.legend()
        
        return fig
    
    # ----- Metrics Calculation -----
    predicted = model_fit.fittedvalues
    actual = ts_values[1:len(predicted)+1]  # Adjust for any differences in length
    
    metrics_df = calculate_metrics(actual, predicted)
    
    @output
    @render.table
    def forecast_metrics():
        """Render the ARIMA forecast metrics table."""
        return metrics_df 