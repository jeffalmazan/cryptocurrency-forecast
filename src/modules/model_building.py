import os
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from math import sqrt
# from read_csv import read_data
# from preprocessing import perform_cleanup
# from pca import apply_pca

# df = read_data()
# df = perform_cleanup(df)
# df = apply_pca(df)


def select_best_model(mae_xgb, mae_lr, mae_rf, mse_xgb, mse_lr, mse_rf, rmse_xgb, rmse_lr, rmse_rf, r2_xgb, r2_lr, r2_rf, xgb_model, lr_model, rf_model):
    # Compares the performance metrics of three models and returns the best-performing model based on MAE.
    model_dict = {
        "XGBoost": {"mae": mae_xgb, "mse": mse_xgb, "rmse": rmse_xgb, "r2": r2_xgb, "model": xgb_model},
        "Linear Regression": {"mae": mae_lr, "mse": mse_lr, "rmse": rmse_lr, "r2": r2_lr, "model": lr_model},
        "Random Forest": {"mae": mae_rf, "mse": mse_rf, "rmse": rmse_rf, "r2": r2_rf, "model": rf_model}
    }
    best_model_name = min(model_dict, key=lambda x: model_dict[x]['mae'])
    best_model_info = model_dict[best_model_name]
    best_model_info['name'] = best_model_name
    return best_model_info

def train_models(df, symbols=None, features=['tradecount', 'PCA_1', 'Volume USDT', 'Crypto Volume'], target_column='Close', test_size=0.1, validation_size=0.1, random_state=42):
    # Trains models for each cryptocurrency symbol in the dataset, then selects and saves the best model for each.

    # Setup directories for saving model files
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_directory, '..', '..')
    models_directory = os.path.join(project_root, 'models')
    os.makedirs(models_directory, exist_ok=True)

    # Process each symbol separately if not specified, use all unique symbols in the dataset
    if symbols is None:
        symbols = df['Symbol'].unique().tolist()

    start_time = time.time()
    best_models = {}

    for symbol in symbols:
        print(f"\nTraining models for {symbol}")
        df_symbol = df[df['Symbol'] == symbol]
        X = df_symbol[features]
        y = df_symbol[target_column]

        # Split data into training, testing, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), random_state=random_state)
        X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=(validation_size / (test_size + validation_size)), random_state=random_state)

        # Setup pipelines for each model type
        xgb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBRegressor())
        ])
        lr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor())
        ])

        # Define parameter grids for hyperparameter tuning
        xgb_param_grid = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__max_depth': [3, 5, 7],
            'xgb__learning_rate': [0.05, 0.1, 0.2],
            'xgb__subsample': [0.8, 0.9, 1.0]
        }
        lr_param_grid = {'lr__fit_intercept': [True, False]}
        rf_param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [3, 5, 7],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        # Perform grid search to find the best hyperparameters for each model
        xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='neg_mean_absolute_error')
        lr_grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='neg_mean_absolute_error')
        rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_absolute_error')

        xgb_grid_search.fit(X_train, y_train)
        lr_grid_search.fit(X_train, y_train)
        rf_grid_search.fit(X_train, y_train)

        # Evaluate models and select the best one
        mae_xgb = mean_absolute_error(y_test, xgb_grid_search.predict(X_test))
        mse_xgb = mean_squared_error(y_test, xgb_grid_search.predict(X_test))
        rmse_xgb = sqrt(mse_xgb)
        r2_xgb = r2_score(y_test, xgb_grid_search.predict(X_test))

        mae_lr = mean_absolute_error(y_test, lr_grid_search.predict(X_test))
        mse_lr = mean_squared_error(y_test, lr_grid_search.predict(X_test))
        rmse_lr = sqrt(mse_lr)
        r2_lr = r2_score(y_test, lr_grid_search.predict(X_test))

        mae_rf = mean_absolute_error(y_test, rf_grid_search.predict(X_test))
        mse_rf = mean_squared_error(y_test, rf_grid_search.predict(X_test))
        rmse_rf = sqrt(mse_rf)
        r2_rf = r2_score(y_test, rf_grid_search.predict(X_test))

        best_model_info = select_best_model(
            mae_xgb, mae_lr, mae_rf,
            mse_xgb, mse_lr, mse_rf,
            rmse_xgb, rmse_lr, rmse_rf,
            r2_xgb, r2_lr, r2_rf,
            xgb_grid_search.best_estimator_,
            lr_grid_search.best_estimator_,
            rf_grid_search.best_estimator_
        )

        best_models[symbol] = best_model_info
        print(f"{symbol}: Best Model: {best_model_info['name']}, MAE: {best_model_info['mae']}, MSE: {best_model_info['mse']}, RMSE: {best_model_info['rmse']}, R2 Score: {best_model_info['r2']}")

        # Save the best model to a file for Flask
        model_filename = os.path.join(models_directory, f'best_model_{symbol}.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model_info['model'], f)
        print(f"Saved {symbol}'s best model to {model_filename}")

    runtime = time.time() - start_time
    print("\nTraining complete.")
    print(f"Total Runtime: {runtime:.2f} seconds")

    return best_models



# best_models = train_models(df)
