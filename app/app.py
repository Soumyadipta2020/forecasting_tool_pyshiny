#!/usr/bin/env python3
from pathlib import Path
import sys

# Prevent writing .pyc files during runs
sys.dont_write_bytecode = True
"""
Main application module for the AI Forecasting Application.

This is the entry point for the Shiny application that connects the UI and server components.
"""

# ===== IMPORTS =====
from shiny import App
from ui_scripts.ui_main import app_ui
from server import server_function

# ===== APP CREATION =====
# Create the app with the single consolidated server function.
APP_DIR = Path(__file__).resolve().parent
app = App(app_ui, server_function, static_assets=str(APP_DIR / "www"))
