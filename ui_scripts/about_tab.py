from shiny import ui

def about_tab():
    return ui.nav_panel(
        "About",
        ui.div(
            ui.h3("About"),
            ui.p("This is an AI-powered forecasting application built with PyShiny."),
            ui.p("Author: Soumyadipta Das"),
            ui.p("Version: 0.0.1"),
            class_="card p-3"
        )
    )
