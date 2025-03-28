#!/usr/bin/env python3
"""
Server-side logic for the AI Forecasting Application.

This module defines the server functions that handle reactive data flow,
data processing, and forecasting models.
"""

# ===== IMPORTS =====
from shiny import reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from global_helpers import calculate_metrics
import seaborn as sns

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
    
    # ----- Data Visualization -----
    @output
    @render.plot
    def data_viz():
        """Render a visualization of the uploaded data."""
        if data.get() is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5, 0.5,
                "Upload data to see visualization",
                ha='center', va='center',
                transform=ax.transAxes
            )
            return fig
            
        df = data.get()
        # Use the first numeric column for visualization
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # If we have a datetime column, use it for x-axis
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                ax.plot(df[date_col], df[target_col], marker='o', linestyle='-', alpha=0.7)
                ax.set_xlabel(date_col)
            else:
                # Otherwise use index
                ax.plot(df.index, df[target_col], marker='o', linestyle='-', alpha=0.7)
                ax.set_xlabel('Index')
                
            ax.set_ylabel(target_col)
            ax.set_title(f'Time Series Plot of {target_col}')
            ax.grid(True, alpha=0.3)
            
            # Add a trend line
            try:
                z = np.polyfit(range(len(df)), df[target_col], 1)
                p = np.poly1d(z)
                ax.plot(range(len(df)), p(range(len(df))), "r--", alpha=0.7, label='Trend')
                ax.legend()
            except:
                pass
                
            plt.tight_layout()
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5, 0.5,
                "No numeric columns found for visualization",
                ha='center', va='center',
                transform=ax.transAxes
            )
            return fig
    
    # ----- Summary Statistics -----
    @output
    @render.table
    def summary_stats():
        """Display summary statistics for the data."""
        if data.get() is None:
            return pd.DataFrame({'Note': ['Upload data to see summary statistics']})
            
        df = data.get()
        # Calculate summary statistics for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return pd.DataFrame({'Note': ['No numeric columns found in the data']})
            
        # Calculate basic statistics
        stats = numeric_df.describe().T
        
        # Add additional statistics
        stats['skew'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        
        # Round to 2 decimal places
        stats = stats.round(2)
        
        # Reset index to make column names a column
        stats = stats.reset_index().rename(columns={'index': 'variable'})
        
        return stats
    
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
        # Get time and target columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        time_var = date_cols[0] if date_cols else df.columns[0]
        target_var = numeric_cols[0] if numeric_cols else df.columns[1]
        
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
    
    # ----- Template Download Handler -----
    @output
    @render.download(filename="sample_data.csv")
    def download_template():
        """Handle template file download."""
        if not input.download_template:
            return None
            
        def generate_content():
            with open("www/timeseries_demo.csv", "r") as file:
                return file.read()
                
        return generate_content

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
        
        # Set a more modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot actual values
        ax.plot(prophet_df['ds'], prophet_df['y'], 'b-', linewidth=2, alpha=0.8, label='Actual')
        
        # Plot forecast
        ax.plot(forecast['ds'].tail(horizon), forecast['yhat'].tail(horizon), 
                'r-', linewidth=2, alpha=0.8, label='Forecast')
        
        # Plot prediction intervals
        ax.fill_between(
            forecast['ds'].tail(horizon+10),
            forecast['yhat_lower'].tail(horizon+10),
            forecast['yhat_upper'].tail(horizon+10),
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        
        ax.set_title(f'Prophet Forecast for {target_var} (Next {horizon} periods)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(target_var, fontsize=12)
        ax.legend(fontsize=10)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
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
        
        # Set a more modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Get date values if available
        if pd.api.types.is_datetime64_any_dtype(ts_data[time_var]):
            x_actual = ts_data[time_var]
            # Generate future dates
            last_date = x_actual.iloc[-1]
            date_range = pd.date_range(start=last_date, periods=horizon+1)[1:]
            x_forecast = date_range
            
            # Plot with dates
            ax.plot(x_actual, ts_values, 'b-', linewidth=2, alpha=0.8, label='Actual')
            ax.plot(x_forecast, forecast_values, 'r-', linewidth=2, alpha=0.8, label='Forecast')
        else:
            # Plot with indices
            ax.plot(range(len(ts_values)), ts_values, 'b-', linewidth=2, alpha=0.8, label='Actual')
            ax.plot(
                range(len(ts_values), len(ts_values) + horizon),
                forecast_values,
                'r-', linewidth=2, alpha=0.8,
                label='Forecast'
            )
        
        ax.set_title(f'ARIMA Forecast for {target_var} (Next {horizon} periods)', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(target_var, fontsize=12)
        ax.legend(fontsize=10)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
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