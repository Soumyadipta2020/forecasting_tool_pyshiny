#!/usr/bin/env python3
"""
Main application module for the AI Forecasting Application.

This is the entry point for the Shiny application that connects the UI and server components.
"""

from pathlib import Path
import sys

from shiny import App
from ui_scripts.ui_main import app_ui
from server import server_function

sys.dont_write_bytecode = True

APP_DIR = Path(__file__).resolve().parent
app = App(app_ui, server_function, static_assets=APP_DIR / "www")
