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
                        ui.input_selectize(
                            "model",
                            "Time Series Model(s)",
                            choices=[
                                "Naive",
                                "Seasonal Naive",
                                "Moving Average",
                                "Drift",
                                "ARIMA",
                                "SARIMA",
                                "ETS",
                                "State Space ARIMA",
                                "Prophet",
                                "GRNN",
                                "ARFIMA",
                                "ARCH",
                                "GARCH",
                                "Neural Network",
                                "AutoML",
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
                            "Predictor Variables",
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_selectize(
                            "model1",
                            "Tabular Model(s)",
                            choices=[
                                "Linear Regression",
                                "GLM",
                                "LASSO",
                                "Ridge Regression",
                                "Logistic Regression",
                            ],
                            selected=["Linear Regression"],
                            multiple=True,
                        ),
                    ),
                    ui.tags.details(
                        ui.tags.summary(
                            ui.span("Advanced settings"),
                            ui.tags.i(class_="fa-solid fa-chevron-down"),
                        ),
                        ui.div(
                            ui.input_checkbox("preprocess_interpolate", "Impute Missing Data (Interpolate)", value=False),
                            ui.input_checkbox("preprocess_outliers", "Cap Outliers (IQR)", value=False),
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
                            ),
                            ui.panel_conditional(
                                "input.data_type === 'Non-Time Series'",
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
                            class_="advanced-settings-body",
                        ),
                        class_="advanced-forecast-settings",
                    ),
                    ui.output_ui("report_preview"),
                    ui.output_ui("forecast_action_controls"),
                    class_="forecast-control-card",
                ),
                ui.div(
                    ui.navset_card_tab(
                        ui.nav_panel(
                            "Results",
                            ui.output_ui("forecast_status"),
                            ui.output_ui("model_plot_filter"),
                            ui.output_ui("forecast_plot_container"),
                            ui.output_ui("forecast_quality_verdict"),
                            ui.output_ui("model_accuracy"),
                        ),
                        ui.nav_panel(
                            "Model Summary",
                            ui.output_text_verbatim("fitted_model"),
                        ),
                        ui.nav_panel(
                            "Validation",
                            ui.panel_conditional(
                                "input.data_type === 'Time Series'",
                                output_widget("backtesting_plot", height="360px", fill=False, fillable=False),
                            ),
                            ui.panel_conditional(
                                "input.data_type === 'Non-Time Series'",
                                ui.output_ui("validation_summary"),
                            ),
                        ),
                        ui.nav_panel(
                            "Diagnostics",
                            ui.panel_conditional(
                                "input.data_type === 'Time Series'",
                                ui.output_ui("residual_diagnostics"),
                                output_widget("residual_diagnostics_plot", height="340px", fill=False, fillable=False),
                                ui.output_ui("anomaly_report"),
                                output_widget("anomaly_plot", height="320px", fill=False, fillable=False),
                            ),
                            ui.panel_conditional(
                                "input.data_type === 'Non-Time Series'",
                                ui.div("Diagnostics are available for time-series forecasts.", class_="alert alert-info py-2"),
                            ),
                        ),
                        ui.nav_panel(
                            "Explainability",
                            ui.panel_conditional(
                                "input.data_type === 'Time Series'",
                                output_widget("forecast_explainability_plot", height="340px", fill=False, fillable=False),
                            ),
                            ui.panel_conditional(
                                "input.data_type === 'Non-Time Series'",
                                output_widget("explainability_plot", height="340px", fill=False, fillable=False),
                            ),
                        ),
                        ui.nav_panel(
                            "Data",
                            ui.input_checkbox("compact_forecast_table", "Compact table", value=True),
                            ui.output_data_frame("forecast_table"),
                        ),
                        title=ui.span(_icon("chart-line"), " Forecast Workspace"),
                        id="forecast_workspace",
                        selected="Results",
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
