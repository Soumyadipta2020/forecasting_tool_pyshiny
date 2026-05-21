#!/usr/bin/env python3
"""
Run script for the AI Forecasting Application.

This is the entry point script that launches the application.
Execute this file to start the PyShiny server and application.
"""

import sys

from app import app

sys.dont_write_bytecode = True

if __name__ == "__main__":
    app.run()
