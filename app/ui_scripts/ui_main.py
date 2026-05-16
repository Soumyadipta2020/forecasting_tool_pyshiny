#!/usr/bin/env python3
"""
Navbar based PyShiny UI for the AI Forecasting Application.
"""

from shiny import ui

from ui_scripts.components.about_tab import about_tab
from ui_scripts.components.data_tab import data_tab
from ui_scripts.components.forecast_tab import forecast_tab
from ui_scripts.components.summary_tab import summary_tab


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


app_ui = ui.page_navbar(
    data_tab(),
    summary_tab(),
    forecast_tab(),
    about_tab(),
    ui.nav_spacer(),
    ui.nav_control(ui.input_dark_mode(mode="light")),
    title=ui.span(
        ui.img(src="brand_logo.png", height="32", class_="brand-logo"),
        "AI Forecasting App ",
        ui.span("v.0.03.4", class_="version-badge"),
    ),
    id="main_nav",
    selected="data",
    navbar_options=ui.navbar_options(bg="#0076d7", theme="dark", underline=True),
    window_title="AI Forecasting App",
    footer=ui.div(
        ui.span(ui.img(src="brand_logo.png", height="22"), " (c) 2023"),
        ui.a("Soumyadipta Das", href="https://sites.google.com/view/soumyadipta-das", target="_blank"),
        class_="app-footer",
    ),
)
