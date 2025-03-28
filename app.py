#!/usr/bin/env python3
"""
Main application module for the AI Forecasting Application.

This is the entry point for the Shiny application that connects the UI and server components.
"""

# ===== IMPORTS =====
from shiny import App
from ui import app_ui
from server import server_function

# ===== APP CREATION =====
# Create and run the app
app = App(app_ui, server_function) 