from shiny import ui


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


def forecast_tab():
    return ui.nav_panel(
        "Forecasting",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "data_type",
                    "Select Data Type",
                    choices=["Time Series", "Non-Time Series"],
                    selected="Time Series",
                ),
                ui.input_select("response_variable", "Select Response Variable", choices=[]),
                ui.input_numeric("horizon", "Forecast Horizon", value=12, min=1),
                ui.panel_conditional(
                    "input.data_type === 'Time Series'",
                    ui.input_checkbox("seasonal", "Is Data Seasonal?", value=False),
                    ui.panel_conditional(
                        "input.seasonal",
                        ui.input_numeric("seasonal_period", "Seasonal Period", value=12, min=1),
                    ),
                    ui.input_select(
                        "model",
                        "Select Time Series Model",
                        choices=[
                            "ARIMA",
                            "SARIMA",
                            "GRNN",
                            "ARFIMA",
                            "ARCH",
                            "GARCH",
                            "Neural Network",
                            "AutoML",
                            "ETS",
                            "Prophet",
                            "State Space ARIMA",
                        ],
                        selected="ARIMA",
                    ),
                ),
                ui.panel_conditional(
                    "input.data_type === 'Non-Time Series'",
                    ui.input_selectize(
                        "x_variables",
                        "Select X Variables",
                        choices=[],
                        multiple=True,
                    ),
                    ui.input_select(
                        "model1",
                        "Select Model for Non-Time Series",
                        choices=[
                            "Linear Regression",
                            "GLM",
                            "Logistic Regression",
                            "LASSO",
                            "Ridge Regression",
                        ],
                        selected="Linear Regression",
                    ),
                ),
                ui.input_action_button(
                    "forecast",
                    "Generate Forecast",
                    icon=_icon("arrow-right"),
                    class_="btn-primary w-100",
                ),
                ui.download_button(
                    "download",
                    "Download",
                    icon=_icon("download"),
                    class_="btn-info w-100 mt-2",
                ),
                width=340,
            ),
            ui.card(
                ui.card_header(_icon("chart-line"), " Forecast Results"),
                ui.output_ui("forecast_status"),
                ui.output_plot("plot", height="470px"),
                ui.output_ui("model_accuracy"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header(_icon("microchip"), " Model Summary"),
                ui.output_text_verbatim("fitted_model"),
                full_screen=True,
            ),
        ),
        value="forecasting",
    )
