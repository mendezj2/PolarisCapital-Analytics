"""Node2Vec graph embeddings."""
import networkx as nx
import numpy as np
try:
    from karateclub import Node2Vec
    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False

def build_similarity_graph(nodes, edges):
    """Build NetworkX graph from nodes and edges."""
    graph = nx.Graph()
    
    for node in nodes:
        node_id = node.get('id', str(node))
        graph.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
    
    for edge in edges:
        source = edge.get('source', edge[0] if isinstance(edge, (list, tuple)) else None)
        target = edge.get('target', edge[1] if isinstance(edge, (list, tuple)) else None)
        weight = edge.get('weight', edge[2] if isinstance(edge, (list, tuple)) and len(edge) > 2 else 1.0)
        
        if source and target:
            graph.add_edge(source, target, weight=weight)
    
    return graph

def run_node2vec(graph, dimensions=64, walk_length=80, num_walks=10):
    """Run Node2Vec embedding."""
    if not KARATECLUB_AVAILABLE or len(graph.nodes()) == 0:
        # Fallback: use degree as simple embedding
        embeddings = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            embeddings[node] = [float(degree)] * dimensions
        return embeddings
    
    try:
        model = Node2Vec(dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
        model.fit(graph)
        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = model.get_embedding()[list(graph.nodes()).index(node)].tolist()
        return embeddings
    except:
        # Fallback
        embeddings = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            embeddings[node] = [float(degree)] * dimensions
        return embeddings

