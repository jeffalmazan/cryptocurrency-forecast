import pandas as pd
import datetime as dt
import os

from modules.read_csv import read_data
from modules.preprocessing import perform_cleanup
from modules.preprocessing import handle_outliers
from modules.visualizations import crypto_visualization
from modules.visualizations import plot_correlation_heatmap
# from modules.xgboost_model import xgboost_model
from modules.bitcoin_model import train_bitcoin



def main():
    # Read CSV
    print("#1 Read Data")
    df = read_data()
    print(df.info())

    # Perform cleanup
    print("#2 Perform cleanup")
    cleaned_df = perform_cleanup(df)

    # Handle Outlier
    # print("#3 Handle Outlier")
    # df_no_outliers = handle_outliers(cleaned_df)

    # Visualization
    print('#4 Visualizations for each Crypto')
    # visualizations_df = crypto_visualization(cleaned_df)
    print('Heatmap')
    heatmap_df = plot_correlation_heatmap(cleaned_df)

    # XGboost
    # print('#5 XGBoost Model')
    # xgboost_df = xgboost_model(cleaned_df)
    
    print('#6 Bitcoin')
    train_bitcoin(heatmap_df)
    

if __name__ == "__main__":
    main()