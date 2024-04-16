import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
# Load model data from the pickle files and visualize actual vs. predicted prices for each cryptocurrency symbol.
def visualize_model_performance(models_directory):

    # List all files in the models directory
    model_files = [f for f in os.listdir(models_directory) if f.startswith('best_model_')]

    for model_filename in model_files:
        full_path = os.path.join(models_directory, model_filename)
        with open(full_path, 'rb') as file:
            model_data = pickle.load(file)

        # Extract the test data, predictions, and dates
        y_test = model_data['y_test']
        predictions = model_data['predictions']
        dates = model_data['dates']  
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions,
            'Date': pd.to_datetime(dates)  # Convert dates to datetime
        })

        # Sort the DataFrame by date
        plot_df.sort_values('Date', inplace=True)

        # Plotting
        symbol = model_filename.replace('best_model_', '').replace('.pkl', '')
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['Date'], plot_df['Actual'], label='Actual Prices', color='blue', linewidth=2)
        plt.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted Prices', color='red', linestyle='--', linewidth=2)
        plt.title(f'Actual vs Predicted Prices for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.gcf().autofmt_xdate()  # Auto-format date labels for better readability
        plt.show()


current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_directory, '..', '..')
models_directory = os.path.join(project_root, 'models')

visualize_model_performance(models_directory)
