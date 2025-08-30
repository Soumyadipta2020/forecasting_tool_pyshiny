#!/usr/bin/env python3
"""
UI component definitions for the AI Forecasting Application.

This module defines the user interface layout and components,
organized by functional sections.
"""

# ===== IMPORTS =====
from shiny import ui
from ui_scripts.components.custom_head import custom_head
from ui_scripts.components.data_tab import data_tab
from ui_scripts.components.summary_tab import summary_tab
from ui_scripts.components.forecast_tab import forecast_tab
from ui_scripts.components.about_tab import about_tab

# ===== UI DEFINITION =====
app_ui = ui.page_fluid(
    custom_head,
    ui.div(
        ui.div(
            ui.input_dark_mode(),
            class_="float-end"
        ),
        class_="clearfix"
    ),
    ui.navset_pill_list(
        data_tab(),
        summary_tab(),
        forecast_tab(),
        about_tab(),
        id="tab",
        widths=(3, 9)
    )
)