import torch.nn as nn
import torch
from torch_scatter import scatter_add
from graphlearn.data import UnionGraph


class GCNLayer(nn.Module):
    """
    A layer of Graph Convolutional Network (GCN) that applies a transformation to the node features
    of a graph based on its topology expressed through the adjacency matrix.

    The layer expects that the input graph will have node features combined in a single tensor,
    and the adjacency relations expressed with an edge index tensor and an optional edge weight tensor.
    
    Attributes:
        in_features (int): Number of features in each input node vector.
        out_features (int): Number of features in each output node vector.
        weight (Tensor): Trainable parameter tensor which transforms node features.
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the GCNConv layer with specified input and output feature dimensions.

        Args:
            in_features (int): The number of features per input node.
            out_features (int): The number of features per output node.
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)  # Initialize the weight matrix with Xavier uniform

    def forward(self, graph: UnionGraph) -> UnionGraph:
        """
        Forward pass of the GCN layer which computes the new feature representation for the nodes in the graph.

        The method performs the following operations:
        - Normalizes node features using the adjacency matrix and optional edge weights.
        - Aggregates features from neighbors using scatter operations.
        - Applies a linear transformation to the aggregated features.

        Args:
            graph (UnionGraph): The graph object containing node features, edge indices,
                                optional edge weights, and other graph-related data.

        Returns:
            torch.Tensor: Output features of nodes after applying the graph convolution.

        Raises:
            AssertionError: If the node features are not stored as a single torch.Tensor.
        """
        assert isinstance(graph.node_features, torch.Tensor), "Node features must be a single torch.Tensor"
        
        x = graph.node_features  # Use the node features tensor directly
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight  # Use the normalized adjacency matrix stored in edge_weight

        # Ensure edge_weight is correctly shaped for broadcasting
        if edge_weight.ndim == 1:
            edge_weight = edge_weight.unsqueeze(-1)  # [num_edges, 1] for broadcasting

        row, col = edge_index
        norm_features = x[col] * edge_weight  # Multiply features by edge weights for normalization
        out = scatter_add(norm_features, row, dim=0, dim_size=x.size(0))  # Aggregate features from neighbors
        out = torch.matmul(out, self.weight)  # Transform features using the weight matrix

        # Update graph with new feature vector
        graph.node_features = out
        return graph




class GCNGraphClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, hidden_layers, num_classes, dropout):
        super(GCNGraphClassifier, self).__init__()
        self.input = nn.Sequential(
            GCNLayer(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        hidden = []
        for _ in range(hidden_layers):
            hidden.append(nn.Sequential(
                GCNLayer(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.hidden = nn.ModuleList(hidden)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), # Normalizes input graph features
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, adj):
        x = self.input((x, adj))
        for h in self.hidden:
            x = h((x, adj))
        # Sum pooling layer
        x = torch.sum(x, dim=1)
        # Classification MLP
        out = self.mlp(x)
        return out
    

class GCNNodeClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim, hidden_layers, num_classes, dropout) -> None:
        super(GCNNodeClassifier, self).__init__()
        self.input = nn.Sequential(
            GCNLayer(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        hidden = []
        for _ in range(hidden_layers):
            hidden.append(nn.Sequential(
                GCNLayer(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.input((x, adj))
        for h in self.hidden:
            x = h((x, adj))
        out = self.out(x)
        return out