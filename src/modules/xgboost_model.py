import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from read_csv import read_data
from preprocessing import perform_cleanup

df = read_data()
df = perform_cleanup(df)

def xgboost_model(df, target_column='Close', features=None, test_size=0.1, validation_size=0.1, random_state=42):
    results = {}
    
    symbols = df['Symbol'].unique()
    if features is None:
        features = ['Open', 'High', 'Low', 'Volume USDT']  # Default features
    
    for symbol in symbols:
        print(f"Training model for {symbol}")
        df_symbol = df[df['Symbol'] == symbol]
        
        X = df_symbol[features]
        y = df_symbol[target_column]
    
        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), random_state=random_state)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=(validation_size / (test_size + validation_size)), random_state=random_state)
    
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 colsample_bytree=0.4,
                                 learning_rate=0.2,
                                 max_depth=7,
                                 alpha=10,
                                 n_estimators=500)
    
        model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], early_stopping_rounds=10, verbose=False)
        y_pred = model.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        
        results[symbol] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'model': model
        }
        
        print(f"Evaluation metrics for {symbol}:")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
    
    return results

results = xgboost_model(df)




