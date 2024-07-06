import networkx as nx
import torch
import numpy as np
from typing import List, Dict, Optional
from graphlearn.data import Graph

def networkx_to_graph(
        G: nx.Graph, directed: bool = False,
        node_feature_names: List[str] = [],
        edge_feature_names: List[str] = [],
        node_feature_dtypes: Optional[Dict[str, torch.dtype]] = None,
        edge_feature_dtypes: Optional[Dict[str, torch.dtype]] = None
    ) -> Graph:
    """
    Converts a NetworkX graph to a Graph dataclass, extracting specified node and edge features.
    Allows specification of data types for each feature.
    Handles both directed and undirected graphs. For undirected graphs, edges are added in both directions.

    Args:
        G (nx.Graph): The NetworkX graph to convert.
        directed (bool): If False, undirected edges are treated as bidirectional.
        node_feature_names (List[str]): Node attributes to extract as features.
        edge_feature_names (List[str]): Edge attributes to extract as features.
        node_feature_dtypes (Optional[Dict[str, torch.dtype]]): Data types for node features.
        edge_feature_dtypes (Optional[Dict[str, torch.dtype]]): Data types for edge features.

    Returns:
        Graph: An instance of the Graph dataclass with node features, edge indices, and edge features.
    """
    # Default data types
    if node_feature_dtypes is None:
        node_feature_dtypes = {}
    if edge_feature_dtypes is None:
        edge_feature_dtypes = {}

    # Function to replace None with np.nan
    def replace_none_with_nan(value):
        return value if value is not None else np.nan

    # Node features
    if node_feature_names:
        node_features = {
            feature: torch.tensor(
                np.array([replace_none_with_nan(G.nodes[node].get(feature, np.nan)) for node in G]), 
                dtype=node_feature_dtypes.get(feature, torch.float32)  # Use specified dtype or default to float32
            )
            for feature in node_feature_names
        }
    else:
        node_features = None

    # Edge index
    if directed or G.is_directed():
        edge_list = list(G.edges())
    else:
        edge_list = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Edge features
    if edge_feature_names:
        edge_features = {
            feature: torch.tensor(
                np.array([replace_none_with_nan(G.edges[edge].get(feature, np.nan)) for edge in edge_list]),
                dtype=edge_feature_dtypes.get(feature, torch.float32)  # Use specified dtype or default to float32
            )
            for feature in edge_feature_names
        }
    else:
        edge_features = None

    return Graph(node_features=node_features, edge_index=edge_index, edge_features=edge_features)
