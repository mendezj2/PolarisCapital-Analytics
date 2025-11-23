"""Embedding/network utilities for astronomy dashboards."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from ml.data_loaders import load_astronomy_data


def get_embedding_network(df=None, method='pca'):
    """
    Generate 2D embeddings for sky network visualization.
    Returns nodes/edges plus method used.
    
    Args:
        df: Optional pandas DataFrame. If None, loads data using load_astronomy_data().
        method: Embedding method ('pca' or 'umap').
    """
    if df is None:
        df = load_astronomy_data()
    if df is None or len(df) == 0:
        return {
            'nodes': [],
            'edges': [],
            'method': method
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        df['feat1'] = np.random.randn(len(df))
        df['feat2'] = np.random.randn(len(df))
        numeric_cols = ['feat1', 'feat2']

    X = df[numeric_cols[:10]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'pca':
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
    else:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        method = 'pca'

    nodes = []
    for i, (x, y) in enumerate(X_2d[:100]):
        node_id = f"star_{i}"
        name = df.iloc[i].get('name', node_id) if i < len(df) else node_id
        properties = {}
        if 'age' in df.columns:
            properties['age'] = float(df.iloc[i]['age']) if i < len(df) else 0.0
        if 'color_index' in df.columns:
            properties['color_index'] = float(df.iloc[i]['color_index']) if i < len(df) else 0.0

        nodes.append({
            'id': node_id,
            'name': str(name),
            'x': float(x),
            'y': float(y),
            'properties': properties
        })

    edges = []
    if len(nodes) > 1:
        coords = np.array([[n['x'], n['y']] for n in nodes])
        distances = cdist(coords, coords)
        k = min(3, len(nodes) - 1)
        for i, node in enumerate(nodes):
            nearest = np.argsort(distances[i])[1:k + 1]
            for j in nearest:
                weight = 1.0 / (1.0 + distances[i][j])
                edges.append({
                    'source': node['id'],
                    'target': nodes[j]['id'],
                    'weight': float(weight)
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'method': method
    }
