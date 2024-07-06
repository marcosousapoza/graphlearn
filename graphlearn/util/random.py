import networkx as nx
import numpy as np

def generate_random_graph(num_nodes: int, num_features: int) -> nx.Graph:
    """
    Generates a random directed graph with specified number of nodes and node features,
    and assigns features directly to nodes and edges within the NetworkX graph structure.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_features (int): Number of features per node.

    Returns:
        nx.DiGraph: A NetworkX DiGraph with node and edge features.
    """
    # Create a random directed graph using NetworkX
    G = nx.gnp_random_graph(num_nodes, 0.5, directed=False)

    # Assign random node features
    for i in G.nodes():
        G.nodes[i]['node_feature'] = np.random.normal(0, 1, num_features)

    # Assign random edge features
    for u, v in G.edges():
        G.edges[u, v]['edge_feature'] = np.random.normal(0, 1, num_features)

    return G


def generate_circular_graph(num_nodes: int, num_features: int) -> nx.Graph:
    """
    Generates a circular graph with specified number of nodes and node features,
    and assigns features directly to nodes and edges within the NetworkX graph structure.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_features (int): Number of features per node.

    Returns:
        nx.Graph: A NetworkX Graph with node and edge features.
    """
    # Create a circular graph using NetworkX
    G = nx.cycle_graph(num_nodes)

    # Assign random node features
    for i in G.nodes():
        G.nodes[i]['node_feature'] = np.random.normal(0, 1, num_features)

    # Assign random edge features
    for u, v in G.edges():
        G.edges[u, v]['edge_feature'] = np.random.normal(0, 1, num_features)

    return G