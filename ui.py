from shiny import ui

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