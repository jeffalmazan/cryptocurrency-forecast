from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import numpy as np
import time

def train_bitcoin(df):
    
    # Record the start time for runtime calculation.
    start_time = time.time()  # Record the start time
    
    df = df[df['Symbol'] == 'BTCUSDT']
    print(df.shape)
    
    features = ['Open', 'High', 'Low', 'Volume USDT'] 
    X = df[features]
    y = df['Close']

    # Step 1: Data Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Model Training Pipeline
    # Pipeline for XGBoost
    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor())
    ])

    # Pipeline for Linear Regression
    lr_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

    # Pipeline for Random Forest
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])


    # Define parameter grids for each model
    xgb_param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.05, 0.1, 0.2],
        'xgb__subsample': [0.8, 0.9, 1.0]
    }

    lr_param_grid = {
        'lr__fit_intercept': [True, False]
    }

    rf_param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [3, 5, 7],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV for each model
    xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='neg_mean_squared_error')
    xgb_grid_search.fit(X_train, y_train)

    lr_grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='neg_mean_squared_error')
    lr_grid_search.fit(X_train, y_train)

    rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_train, y_train)

    # Print best parameters for each model
    print("XGBoost Best Parameters:", xgb_grid_search.best_params_)
    print("Linear Regression Best Parameters:", lr_grid_search.best_params_)
    print("Random Forest Best Parameters:", rf_grid_search.best_params_)

    # Reinitialize the pipelines with best parameters
    best_xgb_pipeline = xgb_grid_search.best_estimator_
    best_lr_pipeline = lr_grid_search.best_estimator_
    best_rf_pipeline = rf_grid_search.best_estimator_

    # Re-train the models with best parameters
    best_xgb_pipeline.fit(X_train, y_train)
    best_lr_pipeline.fit(X_train, y_train)
    best_rf_pipeline.fit(X_train, y_train)

    # Step 4: Model Evaluation
    # Evaluate XGBoost
    y_pred_xgb = best_xgb_pipeline.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print("XGBoost Mean Squared Error:", mse_xgb)
    print("XGBoost RMSE:", np.sqrt(mse_xgb))

    # Evaluate Linear Regression
    y_pred_lr = best_lr_pipeline.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print("Linear Regression Mean Squared Error:", mse_lr)
    print("Linear Regression RMSE:", np.sqrt(mse_lr))

    # Evaluate Random Forest
    y_pred_rf = best_rf_pipeline.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print("Random Forest Mean Squared Error:", mse_rf)
    print("Random Forest RMSE:", np.sqrt(mse_rf))
    
    best_model =  select_best_model(mse_xgb, mse_lr, mse_rf, best_xgb_pipeline, best_lr_pipeline, best_rf_pipeline)
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time  # Calculate the elapsed time
    print("Runtime[Training Bitoin]:", runtime, "seconds\n")  # Print the runtime
    
    return best_model

def select_best_model(mse_xgb, mse_lr, mse_rf, xgb_model, lr_model, rf_model):
    
    models = {
        "XGBoost": xgb_model,
        "Linear Regression": lr_model,
        "Random Forest": rf_model
    }

    best_model_name = min({"XGBoost": mse_xgb, "Linear Regression": mse_lr, "Random Forest": mse_rf}, key={"XGBoost": mse_xgb, "Linear Regression": mse_lr, "Random Forest": mse_rf}.get)
    print('Best model:', best_model_name)
    return models[best_model_name]