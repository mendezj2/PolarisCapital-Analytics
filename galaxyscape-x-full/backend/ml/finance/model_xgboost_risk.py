"""XGBoost risk scoring model."""
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_risk_model(df, feature_cols, target_col='risk_score'):
    """Train risk model."""
    if not XGBOOST_AVAILABLE:
        return {'model': None, 'error': 'XGBoost not installed'}
    
    X = df[feature_cols].fillna(0)
    if target_col not in df.columns:
        # Create mock risk scores
        y = np.random.uniform(0, 100, len(df))
    else:
        y = df[target_col].fillna(df[target_col].median())
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
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
        'features': feature_cols,
        'target': target_col,
        'feature_importance': feature_importance
    }

def predict_risk(model_dict, df, feature_cols):
    """Predict risk scores."""
    if not model_dict or 'model' not in model_dict or model_dict['model'] is None:
        return [50.0] * len(df)
    
    model = model_dict['model']
    X = df[feature_cols].fillna(0)
    predictions = model.predict(X)
    return predictions.tolist()

def calibrate_scores(scores):
    """Calibrate scores to 0-100 range."""
    scores = np.array(scores)
    min_score, max_score = scores.min(), scores.max()
    if max_score > min_score:
        calibrated = ((scores - min_score) / (max_score - min_score)) * 100
    else:
        calibrated = np.full_like(scores, 50.0)
    return calibrated.tolist()

def summarize_metrics(y_true, y_pred):
    """Calculate MAE and RMSE metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {'mae': float(mae), 'rmse': float(rmse)}

