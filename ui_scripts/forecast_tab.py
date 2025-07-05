from shiny import ui

def forecast_tab():
    return ui.nav_panel(
        "Forecast",
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
        ui.div(
            ui.h3("Forecast Results"),
            ui.output_plot("forecast_plot"),
            ui.h3("Forecast Metrics"),
            ui.output_table("forecast_metrics"),
            class_="card p-3 mt-3"
        )
    )
