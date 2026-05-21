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
                    ui.input_checkbox("preprocess_interpolate", "Impute Missing Data (Interpolate)", value=False),
                    ui.input_checkbox("preprocess_outliers", "Cap Outliers (IQR)", value=False),
                    ui.input_numeric("horizon", "Forecast Horizon", value=12, min=1),
                    ui.panel_conditional(
                        "input.data_type === 'Time Series'",
                        ui.input_numeric("scenario_adj", "Future Adjustment (%)", value=0, step=1),
                        ui.input_checkbox("seasonal", "Is Data Seasonal?", value=False),
                        ui.panel_conditional(
                            "input.seasonal",
                            ui.input_numeric("seasonal_period", "Seasonal Period", value=12, min=1),
                        ),
                        ui.input_select(
                            "ts_validation",
                            "Validation Method",
                            choices=["Standard (Train/Test)", "Rolling Cross-Validation"],
                            selected="Standard (Train/Test)",
                        ),
                        ui.input_numeric("ts_test_periods", "Test Periods", value=12, min=1),
                        ui.panel_conditional(
                            "input.ts_validation === 'Rolling Cross-Validation'",
                            ui.input_numeric("rolling_folds", "Rolling Folds", value=4, min=2),
                            ui.input_slider("rolling_initial_pct", "Initial Training %", min=40, max=85, value=60, step=5),
                        ),
                        ui.input_selectize(
                            "model",
                            "Select Time Series Model(s)",
                            choices=[
                                "Naive",
                                "Seasonal Naive",
                                "Moving Average",
                                "Drift",
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
                                "Ensemble",
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
                            ui.input_select(
                                "tabular_split_mode",
                                "Split Method",
                                choices=["Random Split", "Last N Rows"],
                                selected="Random Split",
                            ),
                            ui.input_slider("train_split", "Training Set %", min=50, max=95, value=80, step=5),
                            ui.input_numeric("tabular_test_rows", "Test Rows for Last-N", value=20, min=1),
                        ),
                        ui.input_checkbox("use_scenario", "Enable Scenario Row", value=False),
                        ui.panel_conditional(
                            "input.use_scenario",
                            ui.input_select("scenario_feature", "Scenario Feature", choices=[]),
                            ui.input_text("scenario_value", "Scenario Value", value=""),
                        ),
                    ),
                    ui.input_select(
                        "best_model_metric",
                        "Best Model Metric",
                        choices=["MAPE", "sMAPE", "MASE", "WAPE", "MdAPE", "RMSE", "MAE", "R2", "BIC", "Coverage", "Accuracy"],
                        selected="MASE",
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
                        icon=_icon("file-lines"),
                        class_="btn-secondary w-100 mt-2",
                    ),
                    class_="forecast-control-card",
                ),
                ui.div(
                    ui.card(
                        ui.card_header(_icon("chart-line"), " Forecast Results"),
                        ui.output_ui("forecast_status"),
                        ui.output_ui("model_plot_filter"),
                        ui.output_ui("forecast_plot_container"),
                        ui.output_ui("model_accuracy"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header(_icon("microchip"), " Model Summary"),
                        ui.output_text_verbatim("fitted_model"),
                        full_screen=True,
                    ),
                    ui.panel_conditional(
                        "input.data_type === 'Time Series'",
                        ui.card(
                            ui.card_header(_icon("stethoscope"), " Forecast Diagnostics"),
                            ui.output_ui("residual_diagnostics"),
                            output_widget("residual_diagnostics_plot", height="320px", fill=False, fillable=False),
                            full_screen=True,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.data_type === 'Time Series'",
                        ui.card(
                            ui.card_header(_icon("clock-rotate-left"), " Backtesting"),
                            output_widget("backtesting_plot", height="320px", fill=False, fillable=False),
                            full_screen=True,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.data_type === 'Time Series'",
                        ui.card(
                            ui.card_header(_icon("triangle-exclamation"), " Anomaly Detection"),
                            ui.output_ui("anomaly_report"),
                            output_widget("anomaly_plot", height="300px", fill=False, fillable=False),
                            full_screen=True,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.data_type === 'Time Series'",
                        ui.card(
                            ui.card_header(_icon("lightbulb"), " Forecast Explainability"),
                            output_widget("forecast_explainability_plot", height="300px", fill=False, fillable=False),
                            full_screen=True,
                        ),
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
