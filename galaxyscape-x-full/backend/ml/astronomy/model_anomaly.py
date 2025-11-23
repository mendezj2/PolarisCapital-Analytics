"""Anomaly detection for astronomy."""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_with_isolation_forest(df, feature_cols, contamination=0.05):
    """Isolation Forest anomaly detection."""
    X = df[feature_cols].fillna(0)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = model.score_samples(X)
    
    return {
        'scores': scores.tolist(),
        'labels': labels.tolist(),
        'anomalies': [i for i, label in enumerate(labels) if label == -1]
    }

def detect_with_lof(df, feature_cols, neighbors=20):
    """Local Outlier Factor anomaly detection."""
    X = df[feature_cols].fillna(0)
    
    model = LocalOutlierFactor(n_neighbors=neighbors, contamination=0.05)
    labels = model.fit_predict(X)
    scores = model.negative_outlier_factor_
    
    return {
        'scores': scores.tolist(),
        'labels': labels.tolist(),
        'anomalies': [i for i, label in enumerate(labels) if label == -1]
    }

