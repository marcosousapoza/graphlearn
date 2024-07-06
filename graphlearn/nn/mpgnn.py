import torch
import torch.nn as nn
from graphlearn.nn.pooling import ScatterWrapper
from typing import Union
from graphlearn.data import UnionGraph

class MPGNN(nn.Module):
    """
    Message Passing Graph Neural Network (MPGNN) layer that processes graph data through
    message passing mechanisms. This layer utilizes a multi-layer perceptron (MLP) for message
    generation, a specified aggregation function for pooling messages, and an update MLP to refine 
    node embeddings.

    :param aggregation: The aggregation function name for pooling messages ('sum', 'mean', etc.).
    :type aggregation: str
    :param node_feature_dim: Dimension of the node features.
    :type node_feature_dim: int
    :param edge_feature_dim: Dimension of the edge features, None if no edge features are used.
    :type edge_feature_dim: Union[int, None]
    :param dim_hidden_M: Hidden feature dimension for the message MLP.
    :type dim_hidden_M: int
    :param dim_out_M: Output feature dimension for the message MLP.
    :type dim_out_M: int
    :param num_layers_M: Number of layers in the message MLP.
    :type num_layers_M: int
    :param dim_hidden_U: Hidden feature dimension for the update MLP.
    :type dim_hidden_U: int
    :param dim_out_U: Output feature dimension for the update MLP.
    :type dim_out_U: int
    :param num_layers_U: Number of layers in the update MLP.
    :type num_layers_U: int
    :param dropout_rate: Dropout rate used in both MLPs after ReLU activation (default is 0.5).
    :type dropout_rate: float
    """
    def __init__(
            self, aggregation: str,
            node_feature_dim: int, edge_feature_dim: Union[int, None],
            dim_hidden_M: int, dim_out_M: int, num_layers_M: int,
            dim_hidden_U: int, dim_out_U: int, num_layers_U: int,
            dropout_rate: float = 0.5
        ):
        super(MPGNN, self).__init__()
        dim_in_M = node_feature_dim + (edge_feature_dim if edge_feature_dim is not None else 0)
        self.M = self._build_mlp(dim_in_M, dim_hidden_M, dim_out_M, num_layers_M, dropout_rate)
        dim_in_U = node_feature_dim + dim_out_M
        self.U = self._build_mlp(dim_in_U, dim_hidden_U, dim_out_U, num_layers_U, dropout_rate)
        self.send_message = ScatterWrapper(aggregation)

    def _build_mlp(self, dim_in, dim_hidden, dim_out, num_layers, dropout_rate):
        """
        Helper function to build a Multi-Layer Perceptron (MLP) with specified dimensions and layers.

        :param dim_in: Input dimension to the MLP.
        :type dim_in: int
        :param dim_hidden: Hidden layer dimension in the MLP.
        :type dim_hidden: int
        :param dim_out: Output dimension of the MLP.
        :type dim_out: int
        :param num_layers: Number of layers in the MLP.
        :type num_layers: int
        :param dropout_rate: Dropout rate applied after ReLU activation.
        :type dropout_rate: float
        :return: A sequential model representing the MLP.
        :rtype: nn.Sequential
        """
        layers = [nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(dim_hidden, dim_out))
        return nn.Sequential(*layers)

    def forward(self, graph: UnionGraph):
        """
        Forward pass of the MPGNN, processing input graph data to update node embeddings.

        :param graph: The graph object containing node features, edge indices, and optionally edge features.
        :type graph: UnionGraph
        :return: The graph object with updated node features.
        :rtype: UnionGraph
        """
        x = graph.node_features
        idxE = graph.edge_index
        edge_attr = graph.edge_features if isinstance(graph.edge_features, torch.Tensor) else None
        source, destination = idxE[0], idxE[1]

        # Combine node features and edge features if edge features are available
        message = torch.cat((x[source], edge_attr), dim=-1) if edge_attr is not None else x[source]
        y = self.M(message)
        z = self.send_message((y, destination))
        updated_node_features = self.U(torch.cat((x, z), dim=-1))
        graph.node_features = updated_node_features
        return graph



class VirtualNode(nn.Module):
    """
    Implements a VirtualNode layer as a torch.nn.Module. This layer introduces a virtual node that
    aggregates information across all nodes in the graph and then redistributes it back to enhance
    the representation of each node with global graph context.

    Attributes:
        node_feature_dim (int): Input dimension of node embeddings.
        dim_hidden (int): Hidden dimension for the MLP that processes the virtual node embedding.
        num_layers (int): Number of layers in the MLP.
        dropout_rate (float): Dropout rate used in the MLP.
    """
    def __init__(self, node_feature_dim: int, dim_hidden: int, num_layers: int, dropout_rate: float):
        super(VirtualNode, self).__init__()
        self.mlpV = self._build_mlp(node_feature_dim, dim_hidden, num_layers, dropout_rate)
        self.pool = ScatterWrapper('sum')

    def _build_mlp(self, node_feature_dim, dim_hidden, num_layers, dropout_rate):
        """
        Builds a multi-layer perceptron for processing the virtual node embeddings.

        Args:
            node_feature_dim (int): Input dimension of the embeddings.
            dim_hidden (int): Dimension of the hidden layers in the MLP.
            num_layers (int): Number of layers in the MLP.
            dropout_rate (float): Dropout rate for regularization.
        
        Returns:
            torch.nn.Sequential: The MLP model.
        """
        layers = [nn.Linear(node_feature_dim, dim_hidden), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        return nn.Sequential(*layers)

    def forward(self, graph: UnionGraph) -> UnionGraph:
        """
        Processes the input graph through the VirtualNode layer, adding global context to each node's features.

        Args:
            graph (UnionGraph): The input graph with node embeddings and batch indices.

        Returns:
            UnionGraph: The output graph with updated node embeddings.
        """
        x = graph.node_features
        graph_idx = graph.graph_idx

        # Aggregate global graph embedding using sum pooling
        virtual_node = self.pool(x, graph_idx)
        # Process the global embedding through the MLP
        virtual_node = self.mlpV(virtual_node)
        # Distribute the global context back to each node
        updated_node_features = x + virtual_node[graph_idx]
        # Return a new UnionGraph with updated node features
        return UnionGraph(
            node_features=updated_node_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features if hasattr(graph, 'edge_features') else None,
            graph_idx=graph_idx,
            edge_weight=graph.edge_weight
        )

