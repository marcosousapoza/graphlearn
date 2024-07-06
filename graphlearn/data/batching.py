from typing import List, Optional, Literal, Dict, Union
import graphlearn.data as data
import torch


def _concatenate_features(features: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Concatenate features which may be directly tensors or dictionaries of tensors.
    """
    if len(features) == 0:
        return None
    if isinstance(features[0], dict):
        # Handle dictionaries by concatenating each entry in the dictionary
        concatenated = {key: torch.cat([f[key] for f in features if key in f], dim=0) for key in features[0].keys()}
    else:
        # Handle direct list of tensors
        concatenated = torch.cat(features, dim=0)
    return concatenated


def _split_features(features: Union[torch.Tensor, Dict[str, torch.Tensor]], mask: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Helper function to split features based on a mask.
    :param features: The features to split.
    :param mask: The boolean mask to apply to the features.
    :return: The split features.
    """
    if isinstance(features, dict):
        return {key: val[mask] for key, val in features.items()}
    return features[mask]


def collate(batch: List[data.Graph]) -> data.UnionGraph:
    """
    Combines a batch of Graph instances into a single UnionGraph instance.
    This function concatenates features within the same dimension across all graphs in the batch,
    handling both direct tensors and dictionaries of tensors uniformly.

    :param batch: A list of Graph instances to be collated into a single dataset.
    :type batch: List[data.Graph]
    :return: A single dataset containing the collated features from multiple graphs.
    :rtype: data.UnionGraph
    """
    # Initialize containers
    edge_indices = []
    node_features, edge_features, edge_weights, graph_features, metadata = [], [], [], [], []
    batch_idx = []

    cumsum_node = 0  # Cumulative sum of nodes to adjust edge indices
    for i, graph in enumerate(batch):
        num_nodes = graph.number_of_nodes()

        # Collect features
        if graph.node_features is not None:
            node_features.append(graph.node_features)
        if graph.edge_features is not None:
            edge_features.append(graph.edge_features)
        if graph.edge_weight is not None:
            edge_weights.append(graph.edge_weight)
        if graph.graph_features is not None:
            graph_features.append(graph.graph_features)
        if graph.metadata:
            metadata.append(graph.metadata)

        # Adjust edge indices and collect them
        edge_indices.append(graph.edge_index + cumsum_node)
        
        # Batch indices for nodes
        batch_idx.append(torch.full((num_nodes,), i, dtype=torch.long))
        cumsum_node += num_nodes

    # Concatenate all components
    node_features_collated = _concatenate_features(node_features)
    edge_index_collated = torch.cat(edge_indices, dim=1)
    edge_features_collated = _concatenate_features(edge_features)
    graph_features_collated = _concatenate_features(graph_features)
    metadata_collated = _concatenate_features(metadata)
    edge_weights_collated = _concatenate_features(edge_weights)
    batch_idx_collated = torch.cat(batch_idx, dim=0)

    return data.UnionGraph(
        node_features=node_features_collated,
        edge_index=edge_index_collated,
        edge_weight=edge_weights_collated,
        edge_features=edge_features_collated,
        graph_features=graph_features_collated,
        metadata=metadata_collated,
        graph_idx=batch_idx_collated
    )


def concat(batch: List[data.Graph]) -> data.Graph:
    """
    Combines a batch of Graph instances into a single Gaph instance. 
    This function concatenates features within the same dimension across all graphs in the batch.

    Args:
        batch (List[Graph]): A list of Graph instances to be collated into a single dataset.

    Returns:
        Graph: A single dataset containing the collated features from multiple graphs, with each
                    type of feature being concatenated within its dimension.
    """
    union_graph = collate(batch)

    return data.Graph(
        node_features=union_graph.node_features,
        edge_index=union_graph.edge_index,
        edge_weight=union_graph.edge_weight,
        edge_features=union_graph.edge_features,
        graph_features=union_graph.graph_features,
        metadata=union_graph.metadata
    )

def uncollate(
        union_graph: data.UnionGraph,
        metadata_split_config: Optional[Dict[str, Literal['node', 'edge', 'graph']]] = None
    ) -> List[data.Graph]:
    """
    Splits a UnionGraph instance into a list of individual Graph instances.
    This function effectively reverses the collation process by distributing node, edge, graph features,
    and metadata according to the batch index and specified configuration.

    :param union_graph: A single dataset containing collated features from multiple graphs.
    :type union_graph: data.UnionGraph
    :param metadata_split_config: A dictionary specifying how each metadata key should be distributed across nodes, edges, or the graph.
    :type metadata_split_config: Dict[str, Literal['node', 'edge', 'graph']], optional
    :return: A list of Graph instances, each reconstructed from the collated dataset.
    :rtype: List[data.Graph]
    """
    batch_idx = union_graph.graph_idx
    num_graphs = batch_idx.max().item() + 1

    graphs = []
    for i in range(num_graphs):
        mask_nodes = (batch_idx == i)
        node_features = _split_features(union_graph.node_features, mask_nodes)
        
        mask_edges = (mask_nodes[union_graph.edge_index[0]]) & (mask_nodes[union_graph.edge_index[1]])
        edge_index = union_graph.edge_index[:, mask_edges]
        node_cumulative = mask_nodes.cumsum(0) - 1  # Adjust indices for isolated node features
        edge_index_adjusted = node_cumulative[edge_index]

        edge_features = _split_features(union_graph.edge_features, mask_edges)
        edge_weight = union_graph.edge_weight[mask_edges] if union_graph.edge_weight is not None else None
        graph_features = union_graph.graph_features if union_graph.graph_features is not None else None

        # Handling arbitrary metadata
        metadata = {}

        if metadata_split_config:
            try:
                for key, split_type in metadata_split_config.items():
                    data_to_split = union_graph.metadata[key]
                    if split_type == 'node':
                        metadata[key] = data_to_split[mask_nodes]
                    elif split_type == 'edge':
                        metadata[key] = data_to_split[mask_edges]
                    elif split_type == 'graph':
                        metadata[key] = data_to_split[i] if len(data_to_split.shape) > 0 else data_to_split
                    else:
                        raise ValueError(f'Distribution of type {split_type} is not supported. Use "node", "edge", or "graph"')
            except IndexError:
                raise IndexError('You probably provided an incorrect metadata splitting config.')

        graph = data.Graph(
            node_features=node_features,
            edge_index=edge_index_adjusted,
            edge_features=edge_features,
            edge_weight=edge_weight,
            graph_features=graph_features,
            metadata=metadata
        )
        graphs.append(graph)
    return graphs