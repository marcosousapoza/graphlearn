"""
This submodule provides a framework for working with graph-structured data, facilitating easy integration
and manipulation of graph features within the PyTorch ecosystem. It is designed to streamline operations
on graphs, such as feature extraction, modification, and aggregation, in a manner that is compatible with
common machine learning workflows.

The submodule contains classes and functions tailored for graph data, ensuring that operations specific
to graph features (like node and edge attributes) are handled efficiently. This allows developers to
leverage powerful graph-based algorithms within standard PyTorch models.

Example:
    To demonstrate the use of this submodule within a PyTorch model, consider a simple scenario where we
    want to extract node features from a graph and apply a linear transformation to these features:

    ```python
    from torch import nn, torch
    from graphlearn.data import Graph
    from mymodule.graph_submodule import FeatureExtractor

    # Define the number of nodes
    num_nodes = 10

    # Define a simple graph with node features and self-loops for each node
    node_features = torch.randn(num_nodes, 5)  # 10 nodes, 5 features each
    edge_index = torch.vstack((torch.arange(num_nodes), torch.arange(num_nodes)))  # Self-loops

    example_graph = Graph(node_features=node_features, edge_index=edge_index)

    # Create a PyTorch sequential model that first extracts node features, then applies a linear transformation
    model = nn.Sequential(
        FeatureExtractor('node'),
        nn.Linear(5, 2)  # Transforms the 5-dimensional node features to 2 dimensions
    )

    # Pass the graph through the model to get transformed node features
    transformed_features = model(example_graph)
    ```

In this example, `FeatureExtractor` is used to pull out node features from a `Graph` object, which are then
fed directly into a linear layer for transformation. This modular approach allows for flexible and powerful
architectures specifically tailored to graph-based machine learning models.
"""
from typing import Dict, Literal, List, Union, Optional
from graphlearn.data import Graph, UnionGraph
import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """
    A module designed to extract features (node, edge, or graph) from a Graph or UnionGraph object.
    This feature extractor is specialized to retrieve attributes defined as 'node_features', 'edge_features', 
    or 'graph_features' based on the input specification.

    :param attribute: The type of features to extract ('node', 'edge', or 'graph').
    :type attribute: Literal['node', 'edge', 'graph']
    """
    def __init__(self, attribute: Literal['node', 'edge', 'graph'], key:Optional[str] = None):
        """
        Initializes the FeatureExtractor with the specified feature type to extract.

        :param attribute: Type of the feature to extract, must be one of 'node', 'edge', or 'graph'.
        :type attribute: Literal['node', 'edge', 'graph']
        :param key: Used in case the attribute of Graph is a dictionary
        """
        super(FeatureExtractor, self).__init__()
        self.attribute = attribute
        self.key = key

    def forward(self, batch: Union[Graph, UnionGraph]) -> Union[torch.Tensor, Dict[str, Union[Graph, UnionGraph]]]:
        """
        Forward pass of the FeatureExtractor which retrieves the specified feature type from the provided
        Graph or UnionGraph object.

        :param batch: The graph object from which features will be extracted.
        :type batch: Union[Graph, UnionGraph]

        :return: The extracted features which could be a tensor or a dictionary of string keys to Graph or UnionGraph objects.
        :rtype: Union[torch.Tensor, Dict[str, Union[Graph, UnionGraph]]]
        """
        attr = getattr(batch, f'{self.attribute}_features')
        if self.key:
            return attr[self.key]
        return attr


class FeatureAggregator(nn.Module):
    """
    A module for stacking graph feature tensors directly without applying transformation modules.

    :param feature_keys: A dictionary mapping feature types ('node', 'edge', 'graph') to lists of keys.
    :type feature_keys: Dict[str, list]
    """
    def __init__(self, feature_keys: Dict[str, list]) -> None:
        """
        Initialize the FeatureAggregator with specific keys for the features to be aggregated.

        :param feature_keys: Specific keys for the features to be aggregated, mapping types to lists of keys.
        """
        super(FeatureAggregator, self).__init__()
        self.feature_keys = feature_keys

    def forward(self, batch: Union[Graph, UnionGraph]) -> Union[Graph, UnionGraph]:
        """
        Forward pass to concatenate tensors along the last dimension based on the feature keys.

        :param batch: A graph or union graph with dictionaries of tensors for 'node', 'edge', and 'graph' features.
        :type batch: Union[Graph, UnionGraph]
        :return: The same graph or union graph with updated features where tensors have been concatenated.
        :rtype: Union[Graph, UnionGraph]
        """
        for feature_type in ['node', 'edge', 'graph']:
            if feature_type in self.feature_keys:
                feature_list = [
                    getattr(batch, f"{feature_type}_features")[key] 
                    for key in self.feature_keys[feature_type]\
                        if key in getattr(batch, f"{feature_type}_features")
                ]
                if feature_list:
                    concatenated_features = torch.cat(feature_list, dim=-1)
                    setattr(batch, f"{feature_type}_features", concatenated_features)
        return batch
    

class FeatureProcessor(nn.Module):
    """
    A submodule that applies a specific model to a specific attribute of graph features.

    This module allows for the application of a neural network model to one of the attributes 
    of a graph's features, which can be 'node', 'edge', or 'graph' features. Optionally, if the 
    attribute is a dictionary, a specific key within that dictionary can be targeted.

    :param modle: A neural network module to be applied to the specified graph feature.
    :type modle: nn.Module
    :param attribute: The attribute of the graph features to which the model will be applied.
                      Must be one of 'node', 'edge', or 'graph'.
    :type attribute: Literal['node', 'edge', 'graph']
    :param key: The key within the attribute dictionary to which the model will be applied, if applicable.
    :type key: Optional[str]

    :raises AssertionError: If the attribute is not one of 'node', 'edge', or 'graph'.
    :raises KeyError: If a key is provided but is invalid for the attribute dictionary.

    Example usage:
    --------------
    >>> model = SomeModel()
    >>> processor = FeatureProcessor(model, attribute='node', key='features')
    >>> processed_batch = processor(batch)

    Methods
    -------
    __init__(modle: nn.Module, attribute: Literal['node', 'edge', 'graph'], key: Optional[str] = None)
        Initializes the FeatureProcessor with the specified model, attribute, and optional key.
    
    forward(batch: Union[UnionGraph, Graph]) -> Union[UnionGraph, Graph]
        Applies the model to the specified attribute of the batch and returns the modified batch.
    """
    def __init__(self, modle:nn.Module, attribute:Literal['node', 'edge', 'graph'], key:Optional[str] = None) -> None:
        assert attribute in ['node', 'edge', 'graph'], 'attribute must be on of "node", "edge" or "graph"'
        super(FeatureProcessor, self).__init__()
        self.modle = modle
        self.attribute = attribute
        self.key = key

    def forward(self, batch:Union[UnionGraph, Graph]) -> Union[UnionGraph, Graph]:
        """
        Forward pass of the FeatureProcessor.

        This method applies the specified model to the chosen attribute of the batch's features.
        If a key is provided and the attribute is a dictionary, the model is applied to the specific 
        key within the dictionary. Otherwise, it is applied directly to the tensor attribute.

        :param batch: A graph or union graph with features that will be processed.
        :type batch: Union[UnionGraph, Graph]
        :return: The input batch with modified features.
        :rtype: Union[UnionGraph, Graph]

        :raises KeyError: If the attribute is a dictionary and the provided key is invalid.
        """
        attr = getattr(batch, f'{self.attribute}_features')
        if self.key and isinstance(attr, dict):
            attr[self.key] = self.modle.forward(attr[self.key])
        elif isinstance(attr, torch.Tensor):
            tensor = self.modle.forward(attr)
            setattr(batch, f'{self.attribute}_features', tensor)
        else:
            raise KeyError('The provided key is invalid.')
        return batch