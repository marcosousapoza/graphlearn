from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np
import networkx as nx
from tqdm import tqdm

    
def vectorized_hash(x:np.ndarray[np.int64]) -> np.int64:
    """Hash function for 64 bit integers

    Args:
        x (np.int64): integer

    Returns:
        np.int64: hash
    """
    x = x ^ (x >> 21)
    x = x ^ (x << 24)
    x = (x + (x << 3)) + (x << 8)
    x = x ^ (x >> 14)
    x = (x + (x << 2)) + (x << 4)
    x = x ^ (x >> 28)
    x = x + (x << 31)
    return x


def weisfeilerlehman(graphs:list[nx.Graph], iterations:int, use_labels:bool, norm:str=None) -> np.ndarray:
    """
    Computes the Weisfeiler-Lehman (WL) kernel matrix for a list of graphs using the specified number of iterations 
    for label propagation. This kernel matrix captures the similarity between graphs based on their structure and 
    node labels.

    Parameters:
    - graphs (list[nx.Graph]): A list of NetworkX graph objects to compute the WL kernel for.
    - iterations (int): The number of iterations to run the WL algorithm. Each iteration propagates and hashes
        the labels through the graph structure.
    - use_labels (bool): A boolean flag to determine whether to use existing node labels as initial labels.
        If set to False, all nodes are initialized with the label 1.

    Returns:
    - np.ndarray: A symmetric matrix of shape (len(graphs), len(graphs)) where each element (i, j) represents 
        the kernel between graphs[i] and graphs[j] based on the final node labels obtained after the specified 
        number of iterations.

    Overview:
    The function starts by initializing the node labels for each graph either with a default label of 1 or using
    hash of the 'node_label' attribute of each node if `use_labels` is True. Each graph's adjacency matrix is
    converted to a sparse format and adjusted to include self-connections. Label propagation is performed
    through matrix multiplication across the specified number of iterations, with labels being hashed at each
    step to ensure unique representation. The resulting labels are used to construct histograms for each graph,
    which are then used to compute the gram matrix by calculating the dot product of the histograms matrix with
    its transpose. The gram matrix quantifies the similarity between all pairs of graphs based on their structural
    and labeled characteristics post iterations.

    Note:
    The function assumes that 'node_label' attribute exists for all nodes in each graph if `use_labels` is True.
    The implementation requires sufficient memory to handle dense representations temporarily, particularly for
    large graphs or a high number of iterations. [Docstring written by ChatGPT]
    """
    # Convert to adjecency matrices
    initial_colorings = (
        np.ones(g.number_of_nodes(), dtype=np.int64) # Initialise with ones
        if not use_labels
        else np.array( # initialise with hash of label
            [hash(data['node_label']) for _, data in g.nodes(data=True)],
            dtype=np.int64
        )
        for g in graphs
    )
    adjecency_matrices = (
        nx.to_scipy_sparse_array(g, dtype=np.int64)# + sparse.eye(g.number_of_nodes(), dtype=np.int64)
        for g in graphs
    )
    
    # Get maximum dimension for buffering
    max_dim = max(g.number_of_nodes() for g in graphs)
    
    # Get all the dimensions for convenience
    G, I, C = len(graphs), iterations, max_dim
    
    # Compute colors of neighbors for each iteration
    all_neighbor_colors = []
    for matrix, color in tqdm(zip(adjecency_matrices, initial_colorings),
        total=len(graphs), desc='Computing Kernel'):
        # Go through all iterations
        neighbor_colors = []
        colored_graph:sparse.csr_matrix = matrix
        for _ in range(iterations):
            # compute hash of neighbor colors
            diag_color = sparse.diags(color, dtype=np.int64)
            colored_graph = colored_graph.dot(diag_color)
            colored_graph.data = vectorized_hash(colored_graph.data)
            # add self color to hash
            color = np.array(
                vectorized_hash(colored_graph.sum(axis=1) + color)
            )
            padded_color = np.pad(color, (0, max_dim - len(color)), 'constant')
            neighbor_colors.append(padded_color)
        all_neighbor_colors.append(np.hstack(neighbor_colors))
       
    # Stack and recolor
    all_neighbor_colors = np.stack(all_neighbor_colors, dtype=np.int64)
    uniq, all_neighbor_colors = np.unique(
        all_neighbor_colors.flatten(), return_inverse=True
    )
    
    # Get the index of the buffer variable i.e. 0
    buffer_index = np.where(uniq == 0)[0][0]
    
    # Compute the histogram
    rows = np.repeat(np.arange(G), I*C)
    cols = all_neighbor_colors
    data = np.ones_like(cols)

    # Create sparse matrix and sum duplicates directly
    histograms = sparse.coo_matrix((data, (rows, cols)), shape=(len(graphs), len(uniq))) # -1 because removal of buffere element
    histograms = histograms.tocsr()  # Sums duplicate row column pairs
    
    # Remove buffer index
    if buffer_index.size != 0:
        histograms = histograms[:, np.arange(histograms.shape[1]) != buffer_index]

    # Normalize each feature vector
    if norm:
        histograms = normalize(histograms, norm=norm, axis=1)

    # Compute the gram matrix
    gram_matrix = histograms.dot(histograms.transpose())
    
    return gram_matrix.toarray()
    

def compute_wl(graphs:list[nx.Graph], iterations:int, use_labels:bool):
    """
    1-Weisfeiler Leman Kernel slower than wl_kernel but the same. Computes the gram matrix 
    where the feature vectors of the graphs are the color histograms after the specified iterations 
    of color refinement, 

    Args:
        graphs (list[nx.Graph]): The list of graphs to compute the gram matrix for
        iterations (int): Number of iterations of color refinement.
        use_labels (bool): Use labels for initial coloring.

    Returns:
        :  - np.ndarray: A symmetric matrix of shape (len(graphs), len(graphs)) where each element (i, j) represents 
        the kernel between graphs[i] and graphs[j] based on the final node labels obtained after the specified 
        number of iterations.
    """
    # One feature vector for each graph
    features = []
    
    offset = 0
    graph_indices = []

    for g in graphs:
        graph_indices.append((offset, offset + g.number_of_nodes() - 1))
        offset += g.number_of_nodes()

    # Color of each vertex for each iteration.
    color_map = []
    for i, g in enumerate(graphs):
        color_map.append({})
        for v, data in g.nodes(data=True):
            if(use_labels):
                color_map[i][v] = data['node_label']
            else:
                color_map[i][v] = 0

    c = 1
    while c <= iterations:
        colors = []

        for i, g in enumerate(graphs):
            # Iterate over vertices to update color.
            for v in g.nodes:
                # Colors of the neighbours
                neighbors_colors = []

                # Collect colors of neighbors.
                for w in g.neighbors(v):
                    neighbors_colors.append(color_map[i][w])

                # Sort the colors.
                neighbors_colors.sort()

                # Add color of vertex v itself.
                neighbors_colors.append(color_map[i][v])
                colors.append(hash(tuple(neighbors_colors)))

        # Map to 0 to n interval again.
        _, colors = np.unique(colors, return_inverse=True)

        # Assign new colors to vertices.
        q = 0
        for i, g in enumerate(graphs):            
            for v in g.nodes:
                color_map[i][v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        # Count how often each color occurs in each graph and create color count vector
        feature_vector = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for index in
                           graph_indices]
        features.append(sparse.csr_array(feature_vector))
        c += 1
    
    W = sparse.hstack(features)
    W = W @ W.transpose()
    return W.toarray()
