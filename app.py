#!/usr/bin/env python3
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
import os

# ===== APP CREATION =====
# Create and run the app
app = App(app_ui, server_function, static_assets=os.path.join(os.path.dirname(__file__), "www")) 