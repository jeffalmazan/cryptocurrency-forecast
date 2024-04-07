import os
import pickle
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
# from read_csv import read_data
# from preprocessing import perform_cleanup

# df = read_data()
# df = perform_cleanup(df)

# Function to select and return the best model based on Mean Absolute Error
def select_best_model(mae_xgb, mae_lr, mae_rf, xgb_model, lr_model, rf_model):
    # Compare the MAE of each model and return the model with the smallest error
    model_dict = {
        "XGBoost": {"mae": mae_xgb, "model": xgb_model},
        "Linear Regression": {"mae": mae_lr, "model": lr_model},
        "Random Forest": {"mae": mae_rf, "model": rf_model}
    }
    best_model_name = min(model_dict, key=lambda x: model_dict[x]['mae'])
    best_model_info = model_dict[best_model_name]
    best_model_info['name'] = best_model_name
    return best_model_info

# Main function to train and evaluate models for each cryptocurrency symbol
def train_models(df, symbols=None, features=['Open', 'High', 'Low', 'Volume USDT'], target_column='Close', test_size=0.1, validation_size=0.1, random_state=42):
    # Model will train from the defined features above

    # Get the directory of the current file (__file__ is the path to the current script)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from the current directory to get to the project root
    project_root = os.path.join(current_directory, '..', '..')
    # Define the path to the models directory within the project root
    models_directory = os.path.join(project_root, 'models')
    # Create the models directory if it does not exist
    os.makedirs(models_directory, exist_ok=True)


    if symbols is None:
        symbols = df['Symbol'].unique().tolist()

    start_time = time.time()  # Start timing the model training process
    best_models = {}  # Dictionary to store the best model for each symbol

    for symbol in symbols:
        print(f"\nTraining models for {symbol}")
        df_symbol = df[df['Symbol'] == symbol]
        X = df_symbol[features]
        y = df_symbol[target_column]

        # Splitting the dataset into training and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + validation_size), random_state=random_state)
        X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=(validation_size / (test_size + validation_size)), random_state=random_state)

        # Setting up machine learning pipelines for each type of model
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

        # Hyperparameter grids for GridSearchCV
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

        # Training the models using GridSearchCV to find the best parameters
        xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='neg_mean_absolute_error')
        xgb_grid_search.fit(X_train, y_train)
        lr_grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='neg_mean_absolute_error')
        lr_grid_search.fit(X_train, y_train)
        rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_absolute_error')
        rf_grid_search.fit(X_train, y_train)

        # Print the best parameters for each model
        print(f"{symbol} - XGBoost Best Parameters: {xgb_grid_search.best_params_}")
        print(f"{symbol} - Linear Regression Best Parameters: {lr_grid_search.best_params_}")
        print(f"{symbol} - Random Forest Best Parameters: {rf_grid_search.best_params_}")

        # Calculating MAE for each model
        mae_xgb = mean_absolute_error(y_test, xgb_grid_search.predict(X_test))
        mae_lr = mean_absolute_error(y_test, lr_grid_search.predict(X_test))
        mae_rf = mean_absolute_error(y_test, rf_grid_search.predict(X_test))

        # Select the best model based on MAE
        best_model_info = select_best_model(mae_xgb, mae_lr, mae_rf, xgb_grid_search.best_estimator_, lr_grid_search.best_estimator_, rf_grid_search.best_estimator_)
        best_models[symbol] = best_model_info  # Store the best model info

    # After all models have been trained, save the best models to a single .pkl file
    all_models_filename = os.path.join(models_directory, 'all_best_models.pkl')
    with open(all_models_filename, 'wb') as f:
        pickle.dump(best_models, f)
    print(f"Saved all models to {all_models_filename}")

    runtime = time.time() - start_time
    print("\nTraining complete.")
    print(f"Total Runtime: {runtime:.2f} seconds")

    return best_models


# best_models = train_models(df)

