from shiny import ui


CHANGELOGS = [
    ("2024-09-09", ["Sample data added"]),
    ("2024-09-03", ["Bug fixes & Adjustments"]),
    ("2024-08-12", ["State Space ARIMA, ARFIMA forecasting models added"]),
    ("2024-07-29", ["Prophet, GRNN, Neural Network forecasting model added"]),
    (
        "2024-07-13",
        [
            "Model summary added for fitted models in forecasting tab",
            "Switching between forecast plot and model summary is now possible",
        ],
    ),
    (
        "2024-07-11",
        [
            "Histogram added for summary statistics visualization",
            "Variance, IQR & Standard deviation added for summary statistics",
            "LASSO & Ridge Regression issue resolved",
            "Accuracy metrics added for models",
            "Actuals added in forecast download",
            "Fitted values used as forecast value during actual period",
        ],
    ),
    (
        "2024-07-07",
        [
            "Data info added at homepage",
            "New tab created - Summary Statistics",
            "Boxplot, Violin Plot added",
            "Missing value imputation added",
        ],
    ),
    (
        "2024-07-03",
        [
            "Multimodal AI Chatbot added",
            "Outlier treatment added",
            "Data visualization added",
            "File template & error handling added",
        ],
    ),
]


def _change_card(date: str, items: list[str]):
    return ui.accordion_panel(
        date,
        ui.tags.ul(*(ui.tags.li(item) for item in items), class_="tick-list"),
    )


def about_tab():
    return ui.nav_panel(
        "About",
        ui.card(
            ui.card_header(
                ui.img(src="brand_logo.png", height="42", class_="me-2"),
                "About the App",
            ),
            ui.p(
                ui.tags.b("Github Repository: "),
                ui.a("Github", href="https://github.com/Soumyadipta2020/forecasting_tool", target="_blank"),
            ),
            ui.p(
                "The AI Forecasting Tool provides a comprehensive and user-friendly platform for data analysis. "
                "Users can upload data, visualize it, generate forecasts with multiple model families, "
                "download results, review model accuracy, and use an assistant for analysis support."
            ),
            ui.h4("Top Features"),
            ui.tags.ul(
                ui.tags.li("Upload your data"),
                ui.tags.li("Visualize your data"),
                ui.tags.li("Select desired model"),
                ui.tags.li("Predict and forecast"),
                ui.tags.li("Extract the results"),
                ui.tags.li("Discuss multiple topics with AI"),
            ),
        ),
        ui.h3("Latest Changelogs"),
        ui.accordion(*(_change_card(date, items) for date, items in CHANGELOGS), id="changelog"),
        value="about",
    )
