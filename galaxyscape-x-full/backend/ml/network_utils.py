"""Network science utilities."""
import networkx as nx

def degree_centrality(graph):
    """Compute degree centrality."""
    return nx.degree_centrality(graph)

def clustering_coefficient(graph):
    """Compute clustering coefficients."""
    return nx.clustering(graph)

def betweenness_centrality(graph, normalized=True):
    """Compute betweenness centrality."""
    return nx.betweenness_centrality(graph, normalized=normalized)

def detect_communities(graph):
    """Detect communities using label propagation."""
    try:
        communities = list(nx.algorithms.community.label_propagation_communities(graph))
        return communities
    except:
        return [set(graph.nodes())]

def multilayer_graph(layers):
    """Create multilayer graph."""
    multi = nx.DiGraph()
    for layer_name, layer_graph in layers.items():
        for u, v, data in layer_graph.edges(data=True):
            multi.add_edge((layer_name, u), (layer_name, v), **data)
    return multi

