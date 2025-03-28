#!/usr/bin/env python3
"""
UI component definitions for the AI Forecasting Application.

This module defines the user interface layout and components,
organized by functional sections.
"""

# ===== IMPORTS =====
from shiny import ui

# ===== UI DEFINITION =====
app_ui = ui.page_fluid(
    # ----- Header -----
    ui.h2("AI Forecasting App v.0.0.1"),
    
    # ----- Navigation Tabs -----
    ui.navset_pill_list(
        ui.nav_panel(
            "Home",
            ui.navset_tab(
                ui.nav_panel(
                    "Data",
                    ui.input_file("file", "Upload CSV File", accept=[".csv"]),
                    # ----- Variable Selection -----
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
                    # ----- Data Preview Section -----
                    ui.h3("Data Preview"),
                    ui.output_data_frame("uploaded_data")
                )
            )  
        ),
        ui.nav_panel(
            "Forecast",
            # ----- Forecast Settings -----
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
                choices={
                    "auto_arima": "Auto ARIMA",
                    "prophet": "Prophet"
                }
            ),
            ui.input_action_button(
                "run_forecast",
                "Run Forecast",
                class_="btn-primary"
            ),
            
            # ----- Forecast Results -----
            ui.h3("Forecast Results"),
            ui.output_plot("forecast_plot"),
            ui.h3("Forecast Metrics"),
            ui.output_table("forecast_metrics")
        ),
        ui.nav_panel(
            "About",
            ui.h3("About"),
            ui.p("This is an AI-powered forecasting application built with PyShiny."),
            ui.p("Author: Soumyadipta Das"),
            ui.p("Version: 0.03.3")
        ),
        id="tab",
        widths=(2, 10)
    )
) 