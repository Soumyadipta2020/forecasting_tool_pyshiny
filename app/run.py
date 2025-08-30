#!/usr/bin/env python3
import sys
# Prevent writing .pyc files during runs
sys.dont_write_bytecode = True
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