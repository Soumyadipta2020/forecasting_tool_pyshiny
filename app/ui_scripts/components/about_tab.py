from shiny import ui


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


CHANGELOGS = [
    (
        "2026-05-21",
        [
            "Modern UI implemented",
            "All functionalities implemented",
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
        *(_change_card(date, items) for date, items in CHANGELOGS),
        value="about",
    )
