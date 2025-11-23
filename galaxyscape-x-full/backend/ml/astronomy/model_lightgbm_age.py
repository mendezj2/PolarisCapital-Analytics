"""LightGBM age regression model."""
import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def train_lightgbm_age(df, feature_cols, target_col='stellar_age'):
    """Train LightGBM model for age prediction."""
    if not LIGHTGBM_AVAILABLE:
        return {'model': None, 'error': 'LightGBM not installed'}
    
    X = df[feature_cols].fillna(0)
    if target_col not in df.columns:
        y = np.random.uniform(1, 10, len(df)) * 1e9
    else:
        y = df[target_col].fillna(df[target_col].median())
    
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    model.fit(X, y)
    
    return {
        'model': model,
        'features': feature_cols,
        'target': target_col,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

def predict_ages(model_dict, df, feature_cols):
    """Predict ages using trained LightGBM model."""
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

def compare_models(xgb_metrics, lgbm_metrics):
    """Compare XGBoost and LightGBM metrics."""
    return {
        'mae_delta': lgbm_metrics['mae'] - xgb_metrics['mae'],
        'rmse_delta': lgbm_metrics['rmse'] - xgb_metrics['rmse'],
        'winner_mae': 'lightgbm' if lgbm_metrics['mae'] < xgb_metrics['mae'] else 'xgboost',
        'winner_rmse': 'lightgbm' if lgbm_metrics['rmse'] < xgb_metrics['rmse'] else 'xgboost'
    }

