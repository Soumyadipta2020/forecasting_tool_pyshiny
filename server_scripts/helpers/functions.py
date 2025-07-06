# General helper functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

# Optional imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False

def getmode(v, na=True):
    """
    Get the mode of a vector
    
    Args:
        v (list or array): Input vector
        na (bool): Whether to remove NA values
        
    Returns:
        The mode of the vector
    """
    if na:
        v = [x for x in v if not pd.isna(x)]
    
    if not v:  # If the list is empty after removing NAs
        return None
        
    return stats.mode(v, keepdims=False)[0]

if TENSORFLOW_AVAILABLE:
    def lstm_forecast(ts_data, horizon):
        """
        LSTM forecasting function
        
        Args:
            ts_data (array): Time series data
            horizon (int): Forecast horizon
            
        Returns:
            array: Forecasted values
        """
        # Normalize the data
        mean = np.mean(ts_data)
        std = np.std(ts_data)
        normalized_data = (ts_data - mean) / std
        
        # Split data into train and test sets
        train_data = normalized_data[:-horizon]
        
        # Prepare the training data
        X_train = []
        y_train = []
        
        for i in range(len(train_data) - horizon):
            X_train.append(train_data[i:i+horizon])
            y_train.append(train_data[i+horizon])
        
        X_train = np.array(X_train).reshape(-1, horizon, 1)
        y_train = np.array(y_train)
        
        # Define the LSTM model architecture
        model = Sequential()
        model.add(LSTM(50, input_shape=(horizon, 1)))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = normalized_data[-horizon:].reshape(1, horizon, 1)
        
        forecast_normalized = []
        current_sequence = last_sequence.copy()
        
        for _ in range(horizon):
            # Predict the next value
            next_value = model.predict(current_sequence, verbose=0)[0, 0]
            forecast_normalized.append(next_value)
            
            # Update the sequence for the next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_value
        
        # Denormalize the forecast
        forecast_values = np.array(forecast_normalized) * std + mean
        
        return forecast_values
else:
    def lstm_forecast(ts_data, horizon):
        """
        Placeholder LSTM forecasting function when TensorFlow is not available
        
        Args:
            ts_data (array): Time series data
            horizon (int): Forecast horizon
            
        Returns:
            array: Simple forecasted values (last value repeated)
        """
        print("TensorFlow not available. Using simple forecast instead.")
        # Simple forecast: repeat the last value
        return np.repeat(ts_data[-1], horizon)

if H2O_AVAILABLE:
    def automl_forecast(ts_data, horizon):
        """
        AutoML forecasting function using H2O
        
        Args:
            ts_data (array): Time series data
            horizon (int): Forecast horizon
            
        Returns:
            array: Forecasted values
        """
        # Convert the time series data to a data frame
        data_df = pd.DataFrame({
            'Date': np.arange(len(ts_data)),
            'Value': ts_data
        })
        
        # Initialize the H2O cluster
        h2o.init()
        
        try:
            # Convert the data frame to an H2O frame
            h2o_df = h2o.H2OFrame(data_df)
            
            # Set the target variable
            target = "Value"
            
            # Train AutoML model
            aml = H2OAutoML(max_runtime_secs=300, max_models=10)
            aml.train(x=["Date"], y=target, training_frame=h2o_df)
            
            # Generate predictions for the future horizon
            forecast_df = pd.DataFrame({
                'Date': np.arange(len(ts_data), len(ts_data) + horizon)
            })
            forecast_h2o = h2o.H2OFrame(forecast_df)
            forecast_predictions = aml.leader.predict(forecast_h2o)
            
            # Convert the predictions to a numpy array
            forecast_values = h2o.as_list(forecast_predictions)['predict'].values
            
            return forecast_values
        
        finally:
            # Shut down the H2O cluster
            h2o.shutdown(prompt=False)
else:
    def automl_forecast(ts_data, horizon):
        """
        Placeholder AutoML forecasting function when H2O is not available
        
        Args:
            ts_data (array): Time series data
            horizon (int): Forecast horizon
            
        Returns:
            array: Simple forecasted values (linear trend)
        """
        print("H2O not available. Using simple forecast instead.")
        # Simple forecast: linear trend based on the last few values
        last_values = ts_data[-5:]
        slope = (last_values[-1] - last_values[0]) / 4
        return np.array([last_values[-1] + slope * (i + 1) for i in range(horizon)])

def arfima_forecast(x, h):
    """
    ARFIMA forecasting function
    
    Args:
        x (array): Time series data
        h (int): Forecast horizon
        
    Returns:
        array: Forecasted values
    """
    try:
        # Fit ARIMA model
        model = ARIMA(x, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Generate forecasts
        forecast_values = model_fit.forecast(steps=h)
        
        return forecast_values
    
    except Exception as e:
        print(f"Error in ARFIMA forecast: {e}")
        return np.zeros(h) 
