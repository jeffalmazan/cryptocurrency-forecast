from flask import Flask, jsonify, request
import pandas as pd

import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__)

models_directory = '../models/'

def populate_data_for_prediction():
    columns = ['tradecount', 'PCA_1', 'Volume USDT', 'Crypto Volume']
    df = pd.DataFrame(columns=columns)
    
    record_values = {
        'tradecount': 218314,
        'PCA_1': -0.539601,
        'Volume USDT': 500000,
        'Crypto Volume': 2000
    }
    df.loc[0] = record_values
    
    return df

def load_model(currency):
    model_dict = {
        'ADAUSDT': 'best_model_ADAUSDT.pkl',
        'AVAXUSDT': 'best_model_AVAXUSDT.pkl',
        'BNBUSDT': 'best_model_BNBUSDT.pkl',
        'BTCUSDT': 'best_model_BTCUSDT.pkl',
        'DOGEUSDT': 'best_model_DOGEUSDT.pkl',
        'ETHUSDT': 'best_model_ETHUSDT.pkl',
        'LINKUSDT': 'best_model_LINKUSDT.pkl',
        'SOLUSDT': 'best_model_SOLUSDT.pkl',
        'TRXUSDT': 'best_model_TRXUSDT.pkl',
        'XRPUSDT': 'best_model_XRPUSDT.pkl'
    }
    return model_dict.get(currency)

# http://localhost:8000/crypto/currency/BTCUSDT/open/0.0123
@app.route('/crypto/currency/<string:currency>/open/<float:open_value>', methods=['GET'])
def predict_crypto_value(currency, open_value):
    
    model_name = load_model(currency)
    
    # Construct the path to the model file based on user input
    model_path = os.path.join(models_directory, f'{model_name}')

    # Check if the model file exists
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Load the model from file
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
        currency_model = model_dict.get('model')
    
 
    # Perform prediction using the selected model
    prediction = currency_model.predict(populate_data_for_prediction())
    
    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()}), 200
    
    
if __name__ == '__main__':
    
    # app.run(debug=True)
    app.run(host="localhost", port=8000, debug=True)
    
    
    

