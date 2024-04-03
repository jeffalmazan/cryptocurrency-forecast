import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from sklearn.metrics import mean_squared_error
from read_csv import read_data
from preprocessing import perform_cleanup


df = read_data()
preprocessed_df = perform_cleanup(df)

def predict_close_prices_with_arima(preprocessed_df, symbol, arima_order=(5, 1, 0)):
    # Filter the preprocessed data for the selected cryptocurrency symbol
    symbol_data = preprocessed_df[preprocessed_df['Symbol'] == symbol]

    # Convert 'Date' to datetime and set as index
    if not pd.api.types.is_datetime64_any_dtype(symbol_data.index):
        symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
        symbol_data = symbol_data.set_index('Date')
    
    # Sort the data by date 
    symbol_data = symbol_data.sort_index()
    
    close_prices = symbol_data['Close']
    
    # Fit the ARIMA model on the 'Close' price
    model = ARIMA(close_prices, order=arima_order)
    model_fit = model.fit()
    
    
    forecast = model_fit.forecast(steps=10)  
    
    return forecast

predict_close_prices_with_arima(df, 'BTCUSDT')