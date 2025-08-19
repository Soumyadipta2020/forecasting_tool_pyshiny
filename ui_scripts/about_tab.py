from shiny import ui
from ui_scripts.components.common_ui import nav_panel


def about_tab():
    return nav_panel(
        "About",
        ui.div(
            ui.h3("About"),
            ui.p("This is an AI-powered forecasting application built with PyShiny."),
            ui.p("Author: Soumyadipta Das"),
            ui.p("Version: 0.0.1"),
            class_="card p-3",
        ),
    )
