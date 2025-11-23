"""Finance anomaly detection."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_flow_anomalies(df, feature_cols):
    """Detect anomalies in financial flows."""
    X = df[feature_cols].fillna(0)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(X)
    scores = model.score_samples(X)
    
    return {
        'labels': labels.tolist(),
        'scores': scores.tolist(),
        'anomalies': [i for i, label in enumerate(labels) if label == -1]
    }

def mahalanobis_outliers(df, feature_cols):
    """Compute Mahalanobis distance outliers."""
    X = df[feature_cols].fillna(0).values
    
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    
    try:
        inv_cov = np.linalg.inv(cov)
        diff = X - mean
        mahal_dist = np.sqrt(np.diag(diff @ inv_cov @ diff.T))
        
        # Threshold: chi-square with df=len(feature_cols) at 95% confidence
        threshold = np.percentile(mahal_dist, 95)
        outliers = (mahal_dist > threshold).astype(int)
        
        return pd.Series(outliers)
    except:
        return pd.Series([0] * len(df))

