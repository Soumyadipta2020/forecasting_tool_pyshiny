#!/usr/bin/env python3
"""
Run script for the AI Forecasting Application.

This is the entry point script that launches the application.
Execute this file to start the PyShiny server and application.
"""

# ===== IMPORTS =====
from app import app

# ===== APPLICATION RUNNER =====
if __name__ == "__main__":
    app.run() 