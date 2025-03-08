from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# UI Definition
app_ui = ui.page_fluid(
    ui.h2("AI Forecasting App v.0.03.3"),
    
    # File upload
    ui.h3("Data Input"),
    ui.input_file("file", "Upload CSV File", accept=[".csv"]),
    
    # Target variable selection
    ui.input_select(
        "time_variable",
        "Select Time Variable",
        choices=[]
    ),
    ui.input_select(
        "target_variable",
        "Target Variable",
        choices=[]
    ),
    
    # Data Preview
    ui.h3("Data Preview"),
    ui.output_data_frame("uploaded_data"),
    
    # Forecast Settings
    ui.h3("Forecast Settings"),
    ui.input_numeric(
        "forecast_horizon",
        "Forecast Horizon",
        value=12,
        min=1,
        max=100
    ),
    ui.input_select(
        "forecast_model",
        "Forecast Model",
        {
            "auto_arima": "Auto ARIMA",
            "prophet": "Prophet"
        }
    ),
    ui.input_action_button(
        "run_forecast",
        "Run Forecast",
        class_="btn-primary"
    ),
    
    # Forecast Results
    ui.h3("Forecast Results"),
    ui.output_plot("forecast_plot"),
    ui.h3("Forecast Metrics"),
    ui.output_table("forecast_metrics"),
    
    # About Section
    ui.h3("About"),
    ui.p("This is an AI-powered forecasting application built with PyShiny."),
    ui.p("Author: Soumyadipta Das"),
    ui.p("Version: 0.03.3")
)

# Server function
def server(input, output, session):
    # Store uploaded data
    data = reactive.Value(None)
    
    # Handle file upload
    @reactive.effect
    def _():
        file_info = input.file()
        if file_info and len(file_info) > 0:
            file_path = file_info[0]["datapath"]
            df = pd.read_csv(file_path)
            data.set(df)
            
            # Update time variable dropdown
            ui.update_select(
                "time_variable", 
                choices=df.columns.tolist(),
                selected=df.columns[0] if len(df.columns) > 0 else None
            )
            
            # Update target variable dropdown with numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            ui.update_select(
                "target_variable", 
                choices=numeric_cols,
                selected=numeric_cols[0] if len(numeric_cols) > 0 else None
            )
    
    # Render data table
    @output
    @render.data_frame
    def uploaded_data():
        if data.get() is not None:
            return data.get()
        return pd.DataFrame()
    
    # Run forecast when button is clicked
    @reactive.effect
    def _():
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
            # Prophet model
            prophet_df = ts_data.rename(columns={time_var: 'ds', target_var: 'y'})
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=horizon, freq='D')
            forecast = model.predict(future)
            
            # Create forecast plot
            @output
            @render.plot
            def forecast_plot():
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
            
            # Calculate metrics
            actual = prophet_df['y'].values
            predicted = forecast['yhat'][:len(actual)].values
            
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAPE (%)', 'RMSE', 'MAE'],
                'Value': [f"{mape:.2f}%", f"{rmse:.2f}", f"{mae:.2f}"]
            })
            
            @output
            @render.table
            def forecast_metrics():
                return metrics_df
                
        elif model_type == "auto_arima":
            # Auto ARIMA model
            # Convert time series data
            ts_values = ts_data[target_var].values
            
            # Fit ARIMA model
            model = ARIMA(ts_values, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Make forecast
            forecast_values = model_fit.forecast(steps=horizon)
            
            # Create forecast plot
            @output
            @render.plot
            def forecast_plot():
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
            
            # Calculate metrics
            predicted = model_fit.fittedvalues
            actual = ts_values[1:len(predicted)+1]  # Adjust for any differences in length
            
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            
            metrics_df = pd.DataFrame({
                'Metric': ['MAPE (%)', 'RMSE', 'MAE'],
                'Value': [f"{mape:.2f}%", f"{rmse:.2f}", f"{mae:.2f}"]
            })
            
            @output
            @render.table
            def forecast_metrics():
                return metrics_df
    
    # Initialize empty plot and metrics
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

# Create and run the app
app = App(app_ui, server) 