#!/usr/bin/env python3
"""
Navbar based PyShiny UI for the AI Forecasting Application.
"""

from shiny import ui

from ui_scripts.components.about_tab import about_tab
from ui_scripts.components.custom_head import custom_head
from ui_scripts.components.data_tab import data_tab
from ui_scripts.components.forecast_tab import forecast_tab
from ui_scripts.components.summary_tab import summary_tab


def _icon(name: str):
    return ui.tags.i(class_=f"fa-solid fa-{name}")


app_ui = ui.TagList(
    custom_head,
    ui.page_navbar(
    data_tab(),
    summary_tab(),
    forecast_tab(),
    ui.nav_panel(
        "AI Assistant",
        ui.layout_columns(
            ui.card(
                ui.card_header(_icon("sliders"), " Assistant Settings"),
                ui.input_select(
                    "model_gen",
                    "AI Model",
                    choices=[
                        "Meta-Llama-3.2",
                        "gemini-pro",
                        "HuggingFaceTB",
                        "Phi-3.5-mini",
                        "google-gemma-7b-it",
                        "Mixtral-v0.1",
                        "Mistral-v0.3",
                        "Yi-1.5",
                        "gpt-3.5-turbo",
                    ],
                    selected="Meta-Llama-3.2",
                ),
                ui.input_slider(
                    "temperature",
                    "Temperature",
                    min=0,
                    max=1,
                    value=0.5,
                    step=0.1,
                ),
                ui.input_file(
                    "file_chat",
                    "Upload context (.docx, .pptx)",
                    accept=[".docx", ".pptx"],
                    multiple=False,
                ),
            ),
            ui.card(
                ui.card_header(_icon("comments"), " Chat"),
                ui.div(ui.output_ui("chat_history"), class_="chat-history"),
                ui.input_text_area(
                    "prompt",
                    "",
                    placeholder="Type your message here...",
                    width="100%",
                    rows=4,
                ),
                ui.div(
                    ui.input_action_button(
                        "chat",
                        "Send",
                        icon=ui.tags.i(class_="fa-solid fa-paper-plane"),
                        class_="btn-primary",
                    ),
                    ui.input_action_button(
                        "remove_chatThread",
                        "Clear History",
                        icon=ui.tags.i(class_="fa-solid fa-trash-can"),
                        class_="btn-outline-secondary",
                    ),
                    ui.download_button(
                        "download_chat",
                        "Download Chat",
                        icon=ui.tags.i(class_="fa-solid fa-download"),
                        class_="btn-info",
                    ),
                    class_="button-row",
                ),
            ),
            col_widths=[4, 8],
            gap="1rem",
        ),
        value="assistant",
    ),
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
    ),
)
