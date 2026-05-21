from shiny import ui
from shinywidgets import output_widget


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


def forecast_tab():
    return ui.nav_panel(
        ui.span(_icon("chart-line"), "Forecasting", class_="nav-label"),
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.card_header(_icon("sliders"), " Forecast Controls"),
                    ui.output_ui("model_recommendation"),
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
                        ui.input_selectize(
                            "model",
                            "Select Time Series Model(s)",
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
                            selected=["ARIMA"],
                            multiple=True,
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
                        ui.input_selectize(
                            "model1",
                            "Select Model(s) for Non-Time Series",
                            choices=[
                                "Linear Regression",
                                "GLM",
                                "Logistic Regression",
                                "LASSO",
                                "Ridge Regression",
                            ],
                            selected=["Linear Regression"],
                            multiple=True,
                        ),
                        ui.input_checkbox("use_backtesting", "Enable Train/Test Split", value=False),
                        ui.panel_conditional(
                            "input.use_backtesting",
                            ui.input_slider("train_split", "Training Set %", min=50, max=95, value=80, step=5),
                        ),
                    ),
                    ui.input_select(
                        "best_model_metric",
                        "Best Model Metric",
                        choices=["MAPE", "RMSE", "MAE", "R2", "BIC", "Accuracy"],
                        selected="MAPE",
                    ),
                    ui.input_action_button(
                        "forecast",
                        "Generate Forecast",
                        icon=_icon("arrow-right"),
                        class_="btn-primary w-100",
                    ),
                    ui.download_button(
                        "download",
                        "Download Data",
                        icon=_icon("download"),
                        class_="btn-info w-100 mt-2",
                    ),
                    ui.download_button(
                        "download_report",
                        "Download Full Report",
                        icon=_icon("file-pdf"),
                        class_="btn-secondary w-100 mt-2",
                    ),
                    class_="forecast-control-card",
                ),
                ui.div(
                    ui.card(
                        ui.card_header(_icon("chart-line"), " Forecast Results"),
                        ui.output_ui("forecast_status"),
                        output_widget("plot", height="380px", fill=False, fillable=False),
                        ui.output_ui("model_accuracy"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header(_icon("microchip"), " Model Summary"),
                        ui.output_text_verbatim("fitted_model"),
                        full_screen=True,
                    ),
                    ui.panel_conditional(
                        "input.data_type === 'Non-Time Series'",
                        ui.card(
                            ui.card_header(_icon("lightbulb"), " Explainability (Feature Importance)"),
                            output_widget("explainability_plot", height="300px", fill=False, fillable=False),
                            full_screen=True,
                        ),
                    ),
                    ui.card(
                        ui.card_header(_icon("table"), " Forecast Table"),
                        ui.output_data_frame("forecast_table"),
                        full_screen=True,
                    ),
                    class_="forecast-main-panel",
                ),
                col_widths=[4, 8],
            ),
            class_="forecast-page-grid",
        ),
        value="forecasting",
    )
