from graphlearn.data import Graph, UnionGraph
import torch


def adjacency_matrix(graph: Graph | UnionGraph) -> Graph | UnionGraph:
    """
    Generate an optimized sparse adjacency matrix from a Graph or UnionGraph.

    :param graph: The Graph containing all graphs' data.
    :type graph: Union[Graph, UnionGraph]
    :return: A sparse adjacency matrix in CSR format representing the combined graphs.
    :rtype: Union[Graph, UnionGraph]
    """
    node_indices = graph.edge_index[0]
    node_connections = graph.edge_index[1]
    num_nodes = graph.number_of_nodes()

    # Create the adjacency matrix in COO format
    values = torch.ones(node_indices.size(0), dtype=torch.float32)
    adj_matrix = torch.sparse_coo_tensor(
        torch.vstack((node_indices, node_connections)),
        values, (num_nodes, num_nodes)
    )

    # Calculate the degree matrix D using sparse sum over the rows
    D = torch.sparse.sum(adj_matrix, dim=1).values()

    # Calculate D^(-1/2)
    D_inv_sqrt = 1.0 / torch.sqrt(D)

    # Convert D_inv_sqrt to sparse format for efficient multiplication
    D_inv_sqrt = torch.sparse_coo_tensor(
        torch.vstack((torch.arange(num_nodes), torch.arange(num_nodes))),
        D_inv_sqrt, (num_nodes, num_nodes)
    )

    # Perform symmetric normalization D^(-1/2) * A * D^(-1/2)
    normalized_adj_matrix = torch.sparse.mm(D_inv_sqrt, adj_matrix)
    normalized_adj_matrix = torch.sparse.mm(normalized_adj_matrix, D_inv_sqrt)
    graph.edge_index = normalized_adj_matrix.indices()
    graph.edge_weight = normalized_adj_matrix.values().float()

    return graph
