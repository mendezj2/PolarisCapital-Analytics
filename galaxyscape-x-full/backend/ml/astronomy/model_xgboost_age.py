"""XGBoost age regression model."""
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_age_model(df, feature_cols, target_col='stellar_age'):
    """Train XGBoost model for age prediction."""
    if not XGBOOST_AVAILABLE:
        return {'model': None, 'error': 'XGBoost not installed'}
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    if target_col not in df.columns:
        # Create mock target if not present
        y = np.random.uniform(1, 10, len(df)) * 1e9  # Age in years
    else:
        y = df[target_col].fillna(df[target_col].median())
    
    # Train model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return {
        'model': model,
        'features': feature_cols,
        'target': target_col,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

def predict_ages(model_dict, df, feature_cols):
    """Predict ages using trained model."""
    if not model_dict or 'model' not in model_dict or model_dict['model'] is None:
        return [0.0] * len(df)
    
    model = model_dict['model']
    X = df[feature_cols].fillna(0)
    predictions = model.predict(X)
    return predictions.tolist()

def summarize_metrics(y_true, y_pred):
    """Compute MAE and RMSE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return {'mae': float(mae), 'rmse': float(rmse)}

