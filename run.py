#!/usr/bin/env python3
"""
Run script for the AI Forecasting App
"""
from shiny import run_app
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the app
if __name__ == "__main__":
    run_app("app.py", launch_browser=True) 