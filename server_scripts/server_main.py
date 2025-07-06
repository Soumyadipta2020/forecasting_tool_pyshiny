"""
Main server logic for the AI Forecasting Application.
This module defines the main server_function and delegates to submodules.
"""
from shiny import reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import StringIO
from server_scripts.global_helpers import calculate_metrics
from server_scripts.server_forecast import run_prophet_forecast, run_arima_forecast
from server_scripts.server_data import (
    handle_file_upload, render_uploaded_data, render_data_viz, render_summary_stats,
    render_stats_viz, render_download_summary_stats, render_download_template
)

def server_function(input, output, session):
    """
    Main server function for the Shiny application.
    Handles reactive data flow, user inputs, and output rendering.
    """
    # ----- Reactive Values -----
    data = reactive.Value(None)

    # ----- File Upload Handler -----
    handle_file_upload(input, session, data)

    # ----- Data Preview -----
    render_uploaded_data(output, data)

    # ----- Data Visualization -----
    render_data_viz(output, data, input)

    # ----- Summary Statistics -----
    render_summary_stats(output, data)

    # ----- Summary Statistics Visualization -----
    render_stats_viz(output, data, input)

    # ----- Download Summary Statistics -----
    render_download_summary_stats(output, data)

    # ----- Forecast Runner -----
    @reactive.effect
    def _():
        if not input.run_forecast() or data.get() is None:
            return
        df = data.get()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        time_var = date_cols[0] if date_cols else df.columns[0]
        target_var = numeric_cols[0] if numeric_cols else df.columns[1]
        horizon = input.forecast_horizon()
        model_type = input.forecast_model()
        if not time_var or not target_var or time_var not in df.columns or target_var not in df.columns:
            return
        ts_data = df[[time_var, target_var]].copy().sort_values(by=time_var)
        if model_type == "prophet":
            run_prophet_forecast(ts_data, time_var, target_var, horizon, output)
        elif model_type == "auto_arima":
            run_arima_forecast(ts_data, time_var, target_var, horizon, output)

    # ----- Initial Plot and Metrics -----
    @output
    @render.plot
    def forecast_plot():
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
        return pd.DataFrame({
            'Metric': ['Note'],
            'Value': ["Run a forecast to see metrics"]
        })

    # ----- Template Download Handler -----
    render_download_template(output)
