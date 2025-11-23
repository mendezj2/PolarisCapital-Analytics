"""Finance graph embeddings."""
import networkx as nx
try:
    from karateclub import Node2Vec
    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False

def build_correlation_graph(correlation_matrix, threshold=0.6):
    """Build graph from correlation matrix."""
    graph = nx.Graph()
    
    if isinstance(correlation_matrix, dict):
        for ticker, peers in correlation_matrix.items():
            graph.add_node(ticker)
            for peer, corr in peers.items():
                if abs(corr) >= threshold:
                    graph.add_edge(ticker, peer, weight=abs(corr), sign=float(corr))
    else:
        # Assume it's a pandas DataFrame
        import pandas as pd
        if isinstance(correlation_matrix, pd.DataFrame):
            for i, ticker1 in enumerate(correlation_matrix.index):
                graph.add_node(ticker1)
                for j, ticker2 in enumerate(correlation_matrix.columns):
                    if i != j:
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) >= threshold:
                            graph.add_edge(ticker1, ticker2, weight=abs(corr), sign=float(corr))
    
    return graph

def embed_assets(graph, dimensions=32):
    """Embed assets using Node2Vec."""
    if not KARATECLUB_AVAILABLE or len(graph.nodes()) == 0:
        embeddings = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            embeddings[node] = [float(degree) / max(degree, 1)] * dimensions
        return embeddings
    
    try:
        model = Node2Vec(dimensions=dimensions)
        model.fit(graph)
        embeddings = {}
        for node in graph.nodes():
            idx = list(graph.nodes()).index(node)
            embeddings[node] = model.get_embedding()[idx].tolist()
        return embeddings
    except:
        embeddings = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            embeddings[node] = [float(degree) / max(degree, 1)] * dimensions
        return embeddings

