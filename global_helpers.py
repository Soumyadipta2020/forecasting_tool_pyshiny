#!/usr/bin/env python3
"""
Global helper functions and utilities for the AI Forecasting Application.

This module contains shared functions, imports, and global settings
used across the application components.
"""

# ===== IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

# ===== GLOBAL SETTINGS =====
warnings.filterwarnings('ignore')

# ===== METRICS FUNCTIONS =====
def calculate_metrics(actual, predicted):
    """
    Calculate common forecast evaluation metrics.
    
    Parameters:
    -----------
    actual : array-like
        The actual observed values
    predicted : array-like
        The model's predicted values
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing MAPE, RMSE, and MAE metrics
    """
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    metrics_df = pd.DataFrame({
        'Metric': ['MAPE (%)', 'RMSE', 'MAE'],
        'Value': [f"{mape:.2f}%", f"{rmse:.2f}", f"{mae:.2f}"]
    })
    
    return metrics_df 