import numpy as np
import networkx as nx

def closedwalk(graphs:list[nx.Graph], max_length:int=None) -> np.ndarray:
    """
    Calculate the closed walk kernel of an undirected graph up to a specified maximum length.

    This function computes the closed walk kernel by counting the number of closed walks 
    of each length in the graph. It uses the eigenvalues of the symmetric adjacency matrix 
    of the graph to compute these counts efficiently. The adjacency matrix must be symmetric,
    which is the case for undirected graphs. The histogram of closed walks of 
    different lengths is returned, where the i-th element of the histogram represents 
    the number of closed walks of length i in the graph.

    Parameters:
    - graphs (list[nx.Graph]): The graphs for which the closed walk kernel is computed. This should 
      be a list of NetworkX Graph object, representing an undirected graph.
    - max_length (int, optional): The maximum length of closed walks to compute. If not provided,
      it defaults to the number of nodes in the largest graph, which ensures that all 
      significant walk lengths are included.

    Returns:
    - np.ndarray: An array where the i-th element represents the count of closed walks of length i in the graph.
      The length of this array will be `max_length + 1` to include the count from zero up to `max_length`.
    
    Note:
    - The closed walks of length 0 are considered as the self-loops for each node, 
      which by definition are the nodes themselves. Thus, histogram[0] is equal to the number of nodes.
    - This function assumes the graph is undirected and that the adjacency matrix is symmetric. 
      Non-symmetric matrices could lead to incorrect calculations or errors because `numpy.linalg.eigvalsh`
      is used, which is optimized for symmetric matrices.
    """
    
    if not max_length:
        max_length = max(g.number_of_nodes() for g in graphs)
    
    histograms = np.zeros(shape=(len(graphs), max_length + 1))
    for i, graph in enumerate(graphs):
        # Get the adjacency matrix of graph
        adj = nx.to_numpy_array(graph, dtype=np.int64)
            
        # Comnpute the traces of matrix powers
        eigenvalues = np.linalg.eigvalsh(adj)
        for k in range(max_length + 1):
            histograms[i, k] = np.sum(eigenvalues**k)
            
    # Comput the gram matrix
    gram_matrix = histograms @ histograms.T
    return gram_matrix
