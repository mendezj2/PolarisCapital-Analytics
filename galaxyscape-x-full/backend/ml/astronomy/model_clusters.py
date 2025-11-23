"""Clustering algorithms for astronomy."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

def run_kmeans(embeddings, k=5):
    """K-means clustering."""
    if isinstance(embeddings, pd.DataFrame):
        data = embeddings.values
    else:
        data = np.array(embeddings)
    
    if len(data) < k:
        return [0] * len(data)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels.tolist()

def run_dbscan(embeddings, eps=0.5, min_samples=5):
    """DBSCAN clustering."""
    if isinstance(embeddings, pd.DataFrame):
        data = embeddings.values
    else:
        data = np.array(embeddings)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels.tolist()

def run_hdbscan(embeddings, min_cluster_size=5):
    """HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        return run_dbscan(embeddings, eps=0.5, min_samples=min_cluster_size)
    
    if isinstance(embeddings, pd.DataFrame):
        data = embeddings.values
    else:
        data = np.array(embeddings)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data)
    return labels.tolist()

