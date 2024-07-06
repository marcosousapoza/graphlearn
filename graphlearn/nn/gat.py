import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from graphlearn.data import UnionGraph

class GATLayer(nn.Module):
    """
    A Graph Attention Network (GAT) layer that employs a multi-head attention mechanism to aggregate information from
    neighboring nodes. Each attention head computes separate attention coefficients, which dictate the importance of node features.

    :param in_features: Number of input features per node.
    :type in_features: int
    :param out_features: Number of output features per node for each attention head.
    :type out_features: int
    :param heads: Number of attention heads.
    :type heads: int
    :param dropout_rate: Dropout rate applied to the attention coefficients.
    :type dropout_rate: float
    :param alpha: Negative slope coefficient of the LeakyReLU activation in the attention mechanism.
    :type alpha: float
    :param concat: Whether to concatenate or average the outputs from different heads.
    :type concat: bool
    """
    def __init__(self, in_features:int, out_features:int, heads:int=1, dropout_rate:float=0.6, alpha:float=0.2, concat:bool=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.concat = concat

        # Initialize parameters for each head
        self.W = nn.Parameter(torch.empty(size=(heads, in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(heads, 2 * out_features, 1)))
        gain = torch.nn.init.calculate_gain("leaky_relu", alpha)
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        nn.init.xavier_uniform_(self.a.data, gain=gain)

    def forward(self, graph: UnionGraph) -> UnionGraph:
        """
        Forward pass of the GAT layer, which processes input graph data to compute node features updated via
        an attention mechanism across node neighbors.

        :param graph: The graph object containing node features and structure (edge indices).
        :type graph: UnionGraph
        :return: The same graph object with updated node features after applying the attention mechanism.
        :rtype: UnionGraph
        """
        x = graph.node_features
        edge_index = graph.edge_index

        h_prime = []
        for k in range(self.heads):
            Wh = torch.matmul(x, self.W[k])  # Transform node features (N, out_features)
            # Concatenate features of source and target nodes using edge indices
            Wh_i = Wh[edge_index[0], :]  # Source node features (E, out_features)
            Wh_j = Wh[edge_index[1], :]  # Target node features (E, out_features)
            e_ij = torch.cat([Wh_i, Wh_j], dim=1)  # Concatenate features (E, 2*out_features)
            # Attention mechanism
            e = self.leakyrelu(torch.matmul(e_ij, self.a[k]).squeeze(1))  # (E,)
            attention = F.softmax(e, dim=0)
            attention = self.dropout(attention)  # Apply dropout to the attention coefficients
            # Message passing
            h_prime_head = scatter_add(
                attention.unsqueeze(-1) * Wh_j, edge_index[1],
                dim=0, dim_size=x.size(0)
            )
            h_prime.append(h_prime_head)

        # Concatenate or average over heads
        if self.concat:
            h_prime = torch.cat(h_prime, dim=1)
        else:
            h_prime = torch.stack(h_prime, dim=0).mean(0)

        # Update the graph structure with new node features
        graph.node_features = h_prime
        return graph

