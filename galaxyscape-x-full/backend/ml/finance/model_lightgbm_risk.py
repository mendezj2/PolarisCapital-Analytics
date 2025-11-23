"""LightGBM risk model."""
import pandas as pd
import numpy as np
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def train_lightgbm_risk(df, feature_cols, target_col='risk_score'):
    """Train LightGBM risk model."""
    if not LIGHTGBM_AVAILABLE:
        return {'model': None, 'error': 'LightGBM not installed'}
    
    X = df[feature_cols].fillna(0)
    if target_col not in df.columns:
        y = np.random.uniform(0, 100, len(df))
    else:
        y = df[target_col].fillna(df[target_col].median())
    
    model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        total = sum(importances)
        if total > 0:
            feature_importance = {col: float(imp / total) for col, imp in zip(feature_cols, importances)}
    
    return {
        'model': model,
        'feature_importance': feature_importance
    }

def evaluate_portfolio_risk(predictions, actuals):
    """Evaluate portfolio risk metrics."""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return {'mae': float(mae), 'rmse': float(rmse)}

def predict_risk(model_dict, df, feature_cols):
    """Predict risk scores using LightGBM model."""
    if not model_dict or 'model' not in model_dict or model_dict['model'] is None:
        return [50.0] * len(df)
    
    model = model_dict['model']
    X = df[feature_cols].fillna(0)
    predictions = model.predict(X)
    return predictions.tolist()

def summarize_metrics(y_true, y_pred):
    """Calculate MAE and RMSE metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {'mae': float(mae), 'rmse': float(rmse)}

