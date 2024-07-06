import torch
from torch import Tensor
from typing import Dict, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Tuple, Literal
from torch import Tensor
import torch

@dataclass
class Graph:
    edge_index: Tensor
    node_features: Optional[Union[Dict[str, Tensor], Tensor]] = None
    edge_features: Optional[Union[Dict[str, Tensor], Tensor]] = field(default_factory=dict)
    edge_weight: Optional[Tensor] = None
    graph_features: Optional[Union[Dict[str, Tensor], Tensor]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Tensor]] = field(default_factory=dict)  # Container for arbitrary user data

    def copy(self) -> 'Graph':
        """
        Creates a deep copy of the Graph instance.

        :return: A new Graph instance with copied data.
        :rtype: Graph
        """
        return Graph(
            edge_index=self.edge_index.clone(),
            node_features={k: v.clone() for k, v in self.node_features.items()}\
                if isinstance(self.node_features, dict) else self.node_features.clone()\
                    if self.node_features is not None else None,
            edge_features={k: v.clone() for k, v in self.edge_features.items()}\
                if isinstance(self.edge_features, dict) else self.edge_features.clone()\
                    if self.edge_features is not None else None,
            edge_weight=self.edge_weight.clone() if self.edge_weight is not None else None,
            graph_features={k: v.clone() for k, v in self.graph_features.items()}\
                if isinstance(self.graph_features, dict) else self.graph_features.clone()\
                    if self.graph_features is not None else None,
            metadata={k: v.clone() for k, v in self.metadata.items()} if self.metadata is not None else {}
        )

    def flatten(self) -> 'Graph':
        """
        Flattens the Graph object by converting single-entry dictionaries
        in the attributes to their respective values.

        :return: Self for in-place modification.
        """
        def flatten_dict(d):
            if isinstance(d, dict) and len(d) == 1:
                return next(iter(d.values()))
            return d

        self.node_features = flatten_dict(self.node_features)
        self.edge_features = flatten_dict(self.edge_features)
        self.graph_features = flatten_dict(self.graph_features)
        return self
    
    def pop(self, from_: Literal['node', 'edge', 'graph', 'metadata'], key: str) -> Tensor:
        """
        Pops a feature from the specified feature set and simplifies the dictionary if only one item remains.

        :param from_: Specifies the feature set to pop from (node, edge, graph, or metadata).
        :param key: Key under which the feature is stored.
        :return: The feature tensor.
        :raises KeyError: If the key is not found.
        :raises TypeError: If the features are not stored in a dictionary.
        """
        feature_map = {
            'node': self.node_features,
            'edge': self.edge_features,
            'graph': self.graph_features,
            'metadata': self.metadata
        }

        if isinstance(feature_map[from_], dict):
            popped_feature = feature_map[from_].pop(key, None)
            if popped_feature is None:
                raise KeyError(f"Key '{key}' not found in {from_} features.")
            return popped_feature
        else:
            raise TypeError(f"Features for {from_} are not stored in a dictionary; pop operation is not applicable.")

    def to_metadata(self, from_: Literal['node', 'edge', 'graph', 'metadata'], key: str, new_key: Optional[str] = None) -> 'Graph':
        """
        Moves a feature from the specified attribute to the metadata dictionary using a specified key.

        This method allows users to transfer a feature from the node, edge, graph, or existing metadata attributes
        to the metadata dictionary. The original feature is removed from its current location and stored in the 
        metadata dictionary under a new key if provided, otherwise under the same key. This is a usefull function for storing
        the target feature for example.

        :param from_: Specifies the attribute type from which to move the feature ('node', 'edge', 'graph', or 'metadata').
        :type from_: Literal['node', 'edge', 'graph', 'metadata']
        :param key: The key under which the feature is currently stored in the specified attribute.
        :type key: str
        :param new_key: The key under which the feature will be stored in the metadata dictionary. If not provided, the original key is used.
        :type new_key: str, optional
        :return: The updated Graph instance with the feature moved to the metadata dictionary.
        :rtype: Graph
        """
        new_key = new_key if new_key else key
        self.metadata[new_key] = self.pop(from_, key)
        return self

    def to(self, device: torch.device) -> 'Graph':
        """
        Moves all tensors of the Graph to the specified device (e.g., CPU or GPU).

        :param device: The device to transfer tensors to.
        :return: Self for in-place modification.
        """
        self.node_features = {k: v.to(device) for k, v in self.node_features.items()} if isinstance(self.node_features, dict) else self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_features = {k: v.to(device) for k, v in self.edge_features.items()} if isinstance(self.edge_features, dict) else self.edge_features.to(device) if self.edge_features is not None else None
        self.edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None
        self.graph_features = {k: v.to(device) for k, v in self.graph_features.items()} if isinstance(self.graph_features, dict) else self.graph_features.to(device) if self.graph_features is not None else None
        if self.metadata:
            self.metadata = {k: v.to(device) for k, v in self.metadata.items()}
        return self
    
    def add_self_loop(self) -> 'Graph':
        """
        Adds self-loops to the graph's edge index to ensure each node can attend to itself.

        :return: Self with modified edge_index to include self-loops.
        """
        num_nodes = self.number_of_nodes()
        self_loop_edges = torch.arange(0, num_nodes, dtype=torch.long)
        self_loop_edges = torch.stack((self_loop_edges, self_loop_edges))
        self.edge_index = torch.cat([self.edge_index, self_loop_edges], dim=1)
        return self

    def number_of_nodes(self) -> int:
        """
        Returns the number of nodes in the graph based on the node features.

        :return: Number of nodes.
        """
        return self.node_features.size(0) if isinstance(self.node_features, Tensor) else next(iter(self.node_features.values())).size(0)


@dataclass
class UnionGraph(Graph):
    graph_idx: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))

    def to(self, device: torch.device) -> 'UnionGraph':
        super().to(device)
        self.graph_idx = self.graph_idx.to(device)
        return self
    
    def set_target(self, from_: Literal['node', 'edge', 'graph'], key: str) -> 'UnionGraph':
        super().set_target(from_, key)
        return self
    
    def flatten(self) -> 'UnionGraph':
        super().flatten()
        return self

    def add_self_loop(self) -> 'UnionGraph':
        super().add_self_loop()
        return self
