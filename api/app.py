from flask import Flask, jsonify, request
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__)

def apply_pca(df, scaler, pca):
    # Extracting the relevant features for PCA
    features = ['Open', 'High', 'Low']
    X = df[features]
    
    # Standardizing the features using the pre-fitted scaler
    X_scaled = scaler.transform(X)
    
    # Applying PCA transformation using the pre-fitted pca
    X_pca = pca.transform(X_scaled)
    
    # Creating a DataFrame with the PCA features
    pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    
    # Combining the PCA features with the original DataFrame
    df_with_pca = pd.concat([df, df_pca], axis=1)
    
    return df_with_pca

def populate_data_for_prediction(currency, open_value, high_value, low_value, tradecount, crypto_volume, volume_usdt, scaler, pca):
    columns = ['Unix', 'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Crypto Volume', 'Volume USDT', 'tradecount']
    df = pd.DataFrame(columns=columns)
    init_values = {
        'Unix': None, 
        'Date': None, 
        'Symbol': currency, 
        'Open': open_value, 
        'High': high_value, 
        'Low': low_value, 
        'Close': None, 
        'Crypto Volume': crypto_volume, 
        'Volume USDT': volume_usdt, 
        'tradecount': tradecount
    }
    df.loc[0] = init_values
    
    df_with_pca = apply_pca(df, scaler, pca)
    final_columns = ['tradecount', 'PCA_1', 'Volume USDT', 'Crypto Volume']
    df_final = df_with_pca[final_columns]
    
    return df_final

# http://localhost:8000/crypto/currency/BTCUSDT
@app.route('/crypto/currency/<string:currency>', methods=['GET'])
def derive_latest_values_by_currency(currency):
    
    print('currency:', currency)
    
    file_path = Path.cwd() / 'data' / 'Datasets' / f'Binance_{currency}_d.csv'
    print(file_path)
    df = pd.DataFrame()

    # Use Path's .is_file() method to check if the file exists
    if file_path.is_file():
        # Use the Path object directly to read the CSV file
        df = pd.read_csv(file_path, skiprows=1)
    else:
        print(f"File {file_path} not found!")
    
    # Find columns that contain the substring 'Volume' and consider them as volume columns
    volume_columns = [col for col in df.columns if 'Volume' in col]
    # If there is at least one 'Volume' column, rename the first one found to 'Crypto Volume'
    if volume_columns:
        df.rename(columns={volume_columns[0]: 'Crypto Volume'}, inplace=True)    
    print(df.head(1))
    
    # Get desired columns
    columns_to_return = ['Open', 'High', 'Low', 'tradecount', 'Volume USDT', 'Crypto Volume']
    

    # Create a subset DataFrame with the desired columns
    subset_df = df[columns_to_return].head(1)

    # Convert DataFrame to dictionary (list of dictionaries if more than one row)
    result = subset_df.iloc[0].to_dict()#subset_df.to_dict(orient='records')
       
    return jsonify(result), 200
    
    
url = (
    '/crypto/currency/<string:currency>/open/<float:open_value>/high/<float:high_value>/low/<float:low_value>'
    '/tradecount/<int:tradecount>/crypto_volume/<float:crypto_volume>/volume_usdt/<float:volume_usdt>'
)

# http://localhost:8000/crypto/currency/BTCUSDT/open/52137.68/high/52488.77/low/51677.0/tradecount/1542990/crypto_volume/29534.99432/volume_usdt/1539600521.6007729
@app.route(url, methods=['GET'])
def predict_crypto_value(currency, open_value, high_value, low_value, tradecount, crypto_volume, volume_usdt):
    models_directory = 'models/'
    
    model_path = os.path.join(models_directory, f'best_model_{currency}.pkl')
   
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {currency} not found'}), 404
    
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
        
    currency_model = model_dict['model']
    scaler = model_dict['scaler']
    pca = model_dict['pca']
    
    # Prepare the data for prediction using the loaded scaler and PCA
    df_for_prediction = populate_data_for_prediction(currency, open_value, high_value, low_value, tradecount, crypto_volume, volume_usdt, scaler, pca)
    
    # Ensure your model is expecting a DataFrame with the right shape
    # If it's expecting a 2D array (as most scikit-learn models do), you might need to reshape
    prediction = currency_model.predict(df_for_prediction)
    
    return jsonify({'prediction': prediction.tolist()}), 200

if __name__ == '__main__':
    # Get the port from the environment variable
    port = int(os.environ.get('PORT', 5000))  # Default to port 5000 if PORT isn't set
    app.run(host='0.0.0.0', port=port)

    
    
    

