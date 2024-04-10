import pandas as pd
import datetime as dt
import os

from modules.read_csv import read_data
from modules.preprocessing import perform_cleanup
from modules.preprocessing import handle_outliers
from modules.visualizations import crypto_visualization
from modules.visualizations import plot_correlation_heatmap
from modules.model_building import select_best_model
from modules.model_building import train_models
from modules.pca import apply_pca




def main():
    # Read CSV
    print("#1 Read Data")
    df = read_data()
    print(df.info())

    # Perform cleanup
    print("#2 Perform cleanup")
    cleaned_df = perform_cleanup(df)

    # Perform PCA
    perform_pca_df = apply_pca(cleaned_df)

    # Handle Outlier
    # print("#3 Handle Outlier")
    # df_no_outliers = handle_outliers(cleaned_df)

    # Visualization
    print('#4 Visualizations for each Crypto')
    # visualizations_df = crypto_visualization(cleaned_df)
    print('Heatmap')
    heatmap_df = plot_correlation_heatmap(perform_pca_df)

    # Model Building
    print('#5 Model Building and training')
    train_model = train_models(perform_pca_df)
    
    
    

if __name__ == "__main__":
    main()