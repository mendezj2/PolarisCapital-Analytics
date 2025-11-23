"""
Thin compatibility layer that re-exports astronomy ML helpers.

Each algorithm now lives in its own module for better organization and reuse.
"""
from ml.astronomy.color_period_regression import get_color_period_regression
from ml.astronomy.star_age import get_star_age_predictions
from ml.astronomy.clusters import get_cluster_assignments
from ml.astronomy.anomalies import get_anomaly_scores
from ml.astronomy.embeddings import get_embedding_network

__all__ = [
    'get_color_period_regression',
    'get_star_age_predictions',
    'get_cluster_assignments',
    'get_anomaly_scores',
    'get_embedding_network',
]
