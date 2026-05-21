#!/usr/bin/env python3
"""
Navbar based PyShiny UI for the AI Forecasting Application.
"""

from shiny import ui

from ui_scripts.components.about_tab import about_tab
from ui_scripts.components.data_tab import data_tab
from ui_scripts.components.forecast_tab import forecast_tab
from ui_scripts.components.summary_tab import summary_tab


FONT_AWESOME_CSS = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.link(rel="stylesheet", href=FONT_AWESOME_CSS),
        ui.tags.link(rel="stylesheet", href="custom.css?v=ops-20260521o"),
    ),
    data_tab(),
    summary_tab(),
    forecast_tab(),
    about_tab(),
    title=ui.div(
        ui.div(ui.img(src="brand_logo.png", height="36", class_="brand-logo"), class_="brand-mark"),
        ui.div(
            ui.div("AI Forecasting", class_="brand-title"),
            ui.div("FORECASTING INTELLIGENCE", class_="brand-subtitle"),
            class_="brand-copy",
        ),
        class_="brand-lockup",
    ),
    id="main_nav",
    selected="data",
    navbar_options=ui.navbar_options(bg="#101827", theme="dark", underline=True),
    window_title="AI Forecasting App",
    header=ui.tags.script(src="custom.js"),
    footer=ui.div(
        ui.span(ui.img(src="brand_logo.png", height="22"), " (c) 2023"),
        ui.a("Soumyadipta Das", href="https://sites.google.com/view/soumyadipta-das", target="_blank"),
        class_="app-footer",
    ),
)
