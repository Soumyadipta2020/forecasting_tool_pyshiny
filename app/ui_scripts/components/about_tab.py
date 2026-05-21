from shiny import ui


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


CHANGELOGS = [
    (
        "2026-05-21",
        [
            "Modern UI implemented",
            "Core data upload, visualization, summary statistics, forecasting, and report downloads implemented",
            "Added residual diagnostics, anomaly detection, backtesting plots, and forecast explainability",
            "Added stronger forecast metrics including MASE, WAPE, sMAPE, MdAPE, and interval coverage",
            "Added benchmark forecasters, weighted ensembles, and automatic frequency/seasonality profiling",
        ],
    ),
]


def _change_card(date: str, items: list[str]):
    return ui.div(
        ui.div(date, class_="changelog-date"),
        ui.tags.ul(*(ui.tags.li(item) for item in items), class_="tick-list changelog-items"),
        class_="changelog-card",
    )


def about_tab():
    return ui.nav_panel(
        ui.span(_icon("circle-info"), "About", class_="nav-label"),
        ui.card(
            ui.card_header(
                ui.img(src="brand_logo.png", height="42", class_="me-2"),
                "About the App",
            ),
            ui.p(
                ui.tags.b("Github Repository: "),
                ui.a("Github", href="https://github.com/Soumyadipta2020/forecasting_tool_pyshiny", target="_blank"),
            ),
            ui.p(
                "The Forecasting App is a PyShiny tool for exploratory data analysis and forecast model comparison. "
                "Users can upload CSV data, inspect data quality, review summary statistics, compare statistical "
                "and machine-learning forecasting models, validate forecasts, inspect residual diagnostics, "
                "detect anomalies, and download forecast outputs or full HTML reports."
            ),
            ui.h4("Top Features"),
            ui.tags.ul(
                ui.tags.li("Upload CSV data or use the bundled sample time-series dataset"),
                ui.tags.li("Review data quality checks for missing values, duplicates, outliers, mixed types, and timeline gaps"),
                ui.tags.li("Explore summary statistics with histograms, boxplots, and violin plots"),
                ui.tags.li("Forecast with selected models such as ARIMA, SARIMA, ETS, Prophet, baselines, neural networks, AutoML-style regressors, and ensembles"),
                ui.tags.li("Compare models with holdout or rolling cross-validation metrics"),
                ui.tags.li("Inspect residual diagnostics, backtesting plots, anomaly markers, and forecast explainability"),
                ui.tags.li("Download forecast data, summary statistics, and full HTML reports"),
            ),
        ),
        ui.h3("Latest Changelogs"),
        *(_change_card(date, items) for date, items in CHANGELOGS),
        value="about",
    )
