import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_directory, '..', '..')
models_directory = os.path.join(project_root, 'models')

# List all files in the models directory
model_files = [f for f in os.listdir(models_directory) if f.startswith('best_model_')]

# Load performance metrics for a specific cryptocurrency from model .pkl files stored in the specified directory.
def load_metrics(models_directory, symbol):

    metrics = {'Symbol': symbol, 'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
    
    model_filename = f'best_model_{symbol}.pkl'
    full_path = os.path.join(models_directory, model_filename)
    with open(full_path, 'rb') as file:
        model_data = pickle.load(file)

    # Check if 'metrics' key exists in model_data
    if 'metrics' in model_data:
        metrics['Model'] = ['XGBoost', 'Linear Regression', 'Random Forest']
        metrics['MAE'] = [model_data['metrics']['XGBoost']['MAE'],
                          model_data['metrics']['Linear Regression']['MAE'],
                          model_data['metrics']['Random Forest']['MAE']]
        metrics['MSE'] = [model_data['metrics']['XGBoost']['MSE'],
                          model_data['metrics']['Linear Regression']['MSE'],
                          model_data['metrics']['Random Forest']['MSE']]
        metrics['RMSE'] = [model_data['metrics']['XGBoost']['RMSE'],
                           model_data['metrics']['Linear Regression']['RMSE'],
                           model_data['metrics']['Random Forest']['RMSE']]
        metrics['R2'] = [model_data['metrics']['XGBoost']['R2'],
                         model_data['metrics']['Linear Regression']['R2'],
                         model_data['metrics']['Random Forest']['R2']]
    else:
        print(f"Error: Missing metrics in {model_filename}")

    return pd.DataFrame(metrics)

#     Plot a bar graph of performance metrics comparison for the specified cryptocurrency.
def plot_comparison(metrics_df, symbol):

    metrics_df.set_index('Model', inplace=True)
    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Performance Metrics Comparison for {symbol}')
    plt.ylabel('Metric Value')
    plt.xlabel('Model')
    plt.legend(title="Metrics")
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

# Iterate through all symbols and load metrics
for model_filename in model_files:
    symbol = model_filename.replace('best_model_', '').replace('.pkl', '')
    metrics_df = load_metrics(models_directory, symbol)
    # plot_comparison(metrics_df, symbol)



def display_metrics_for_cryptos(models_directory, symbols):

    for symbol in symbols:
        df = load_metrics(models_directory, symbol)
        print(f"Metrics Comparison for {symbol}:")
        print(df)
        print("\n")  # Adds a newline for better separation between outputs


# Extract the symbols and create a list of symbols
symbols = [filename.replace('best_model_', '').replace('.pkl', '') for filename in model_files]

display_metrics_for_cryptos(models_directory, symbols)


