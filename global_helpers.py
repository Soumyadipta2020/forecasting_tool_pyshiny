import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Add any global variables or helper functions here
def calculate_metrics(actual, predicted):
    """Calculate forecast evaluation metrics"""
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    metrics_df = pd.DataFrame({
        'Metric': ['MAPE (%)', 'RMSE', 'MAE'],
        'Value': [f"{mape:.2f}%", f"{rmse:.2f}", f"{mae:.2f}"]
    })
    
    return metrics_df 