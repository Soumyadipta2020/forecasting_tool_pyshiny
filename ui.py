#!/usr/bin/env python3
"""
UI component definitions for the AI Forecasting Application.

This module defines the user interface layout and components,
organized by functional sections.
"""

# ===== IMPORTS =====
from shiny import ui

# ===== CUSTOM HEAD TAGS =====
custom_head = ui.tags.head(
    # FontAwesome 
    ui.tags.link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ),
    # Custom CSS
    ui.tags.link(
        rel="stylesheet",
        href="custom.css"
    ),
    # jQuery (ensure it's loaded before our custom JS)
    ui.tags.script(
        src="https://code.jquery.com/jquery-3.6.0.min.js"
    ),
    # Custom JavaScript
    ui.tags.script(
        src="custom.js"
    )
)

# ===== UI DEFINITION =====
app_ui = ui.page_fluid(
    custom_head,
    ui.div(
        
        ui.div(
            ui.input_dark_mode(),
            class_="float-end"
        ),
        class_="clearfix"
    ),
    
    # ----- Navigation Tabs -----
    ui.navset_pill_list(
        ui.nav_panel(
            "Home",
            ui.navset_tab(
                ui.nav_panel(
                    "Data",
                    ui.page_fluid(
                        ui.div(
                            ui.h3("Select Data Source"),
                            ui.input_select(
                                "data_source",
                                "Select Data Source",
                                choices=["Upload", "Database", "API"]
                            ),
                            ui.input_file("file", "Upload Your File (.csv supported)", accept=[".csv"]),
                            ui.div(
                                ui.download_button(
                                    "download_template", 
                                    "Download template file",
                                    icon=ui.tags.i(class_="fas fa-download"),
                                    class_="btn-info"
                                ),
                                ui.input_action_button(
                                    "upload_data_btn", 
                                    "Upload data",
                                    icon=ui.tags.i(class_="fas fa-upload"),
                                    class_="btn-primary"
                                ),
                                class_="d-flex justify-content-between my-3"
                            ),
                            ui.div(
                                class_="text-danger small",
                                content="Please click after editing (if needed) the table below"
                            ),
                            ui.div(
                                ui.h3("Edit Data"),
                                ui.output_data_frame("uploaded_data"),
                                class_="card p-3 mt-3"
                            ),
                            ui.div(
                                ui.h3("Quick Visualization"),
                                ui.output_plot("data_viz"),
                                class_="card p-3 mt-3"
                            ),
                            class_="card-body"
                        )
                    ),
                ),
                ui.nav_panel(
                    "Summary Statistics",
                    ui.page_fluid(
                        ui.br(),
                        ui.card(
                            ui.card_header(
                                "Summary Statistics"
                            ),
                            ui.card_body(
                                ui.download_button(
                                    "download_summary_stats",
                                    "Download Summary Statistics",
                                    icon=ui.tags.i(class_="fas fa-download")
                                ),
                                ui.output_table("summary_stats")
                            )
                        ),
                        ui.card(
                            ui.card_header(
                                "Visualization",
                                ui.span(
                                    ui.input_action_button(
                                        "viz_collapse_btn",
                                        "",
                                        icon=ui.tags.i(class_="fas fa-minus"),
                                        class_="btn-sm btn-link text-light"
                                    ),
                                    class_="float-end"
                                )
                            ),
                            ui.card_body(
                                ui.div(
                                    ui.h4("Plot Type"),
                                    ui.input_select(
                                        "plot_type",
                                        "",
                                        choices=[
                                            "Boxplot",
                                            "Violin Plot",
                                            "Histogram"
                                        ],
                                        selected="Violin Plot"
                                    ),
                                    ui.output_plot("stats_viz")
                                )
                            )
                        )
                    )
                ),
                id="home_tabs"
            )  
        ),
        ui.nav_panel(
            "Forecast",
            # ----- Forecast Settings -----
            ui.h3("Forecast Settings"),
            ui.div(
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
                    icon=ui.tags.i(class_="fas fa-play"),
                    class_="btn-primary"
                ),
                class_="card p-3"
            ),
            
            # ----- Forecast Results -----
            ui.div(
                ui.h3("Forecast Results"),
                ui.output_plot("forecast_plot"),
                ui.h3("Forecast Metrics"),
                ui.output_table("forecast_metrics"),
                class_="card p-3 mt-3"
            )
        ),
        ui.nav_panel(
            "About",
            ui.div(
                ui.h3("About"),
                ui.p("This is an AI-powered forecasting application built with PyShiny."),
                ui.p("Author: Soumyadipta Das"),
                ui.p("Version: 0.03.3"),
                class_="card p-3"
            )
        ),
        id="tab",
        widths=(3, 9)
    )
) 