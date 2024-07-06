from typing import Any, Iterator, Tuple, Union
from  torch.utils.data import IterableDataset
import torch
import graphlearn.data as data
import random
import numpy as np


class PQWalkSampler(IterableDataset):
    """
    Sampler for generating random walks influenced by node2vec-like parameters p and q.
    This sampler generates one random walk at a time using the provided p, q biases, focusing on single sub-graphs.
    """

    def __init__(self, graph: Union[data.Graph, data.UnionGraph], p: float, q: float, walk_length: int, neg_samples_length: int):
        """
        Initializes the random walk sampler with parameters for node2vec biases.

        Args:
            graph (data.UnionGraph): Graph in COO format from which to sample walks.
            p (float): Return hyperparameter controlling likelihood of immediately revisiting a node in the walk.
            q (float): In-out parameter controlling the search to differentiate between inward and outward nodes.
            walk_length (int): The number of steps in each random walk.
            neg_samples_length (int): Number of negative samples to generate for each node in the walk.
        """
        super().__init__()
        self.graph = graph
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.neg_samples_length = neg_samples_length
        self.prepare_data()

    def prepare_data(self):
        # Convert graph data from COO to a more usable form for sampling
        self.neighbors = {
            node.item(): list(
                self.graph.edge_index[1][self.graph.edge_index[0] == node].numpy()
            ) 
            for node in self.graph.edge_index[0].unique()
        }
        self.graph_indices = self.graph.graph_idx.numpy()\
            if isinstance(self.graph, data.UnionGraph)\
            else np.zeros(torch.max(self.graph.edge_index).item())
        # Cache nodes by graph index
        nodes = torch.arange(len(self.graph_indices))
        self.nodes_by_graph = {idx: [] for idx in set(self.graph_indices)}
        for node, idx in zip(nodes, self.graph_indices):
            self.nodes_by_graph[idx].append(node.item())

    def __iter__(self) -> Iterator:
        while True:
            yield self.random_walk()
    
    def random_walk(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly select a sub-graph
        selected_graph = random.choice(list(self.nodes_by_graph.keys()))
        
        # Get nodes from the selected sub-graph
        nodes_in_graph = self.nodes_by_graph[selected_graph]
        
        # Choose a starting node
        start_node = random.choice(nodes_in_graph)
        walk = [start_node]

        # Generate the walk
        for _ in range(self.walk_length - 1):
            cur_node = walk[-1]
            cur_neighbors = self.neighbors[cur_node]
            if not cur_neighbors:
                break  # No further steps if no neighbors

            # Choose the next node with p and q biases
            next_node = self.choose_next_node(cur_node, walk[-2] if len(walk) > 1 else cur_node, cur_neighbors)
            walk.append(next_node)

        # Handle negative sampling
        non_walk_nodes = [node for node in nodes_in_graph if node not in walk]
        neg_samples = random.sample(non_walk_nodes, self.neg_samples_length)

        return torch.tensor(walk, dtype=torch.long), torch.tensor(neg_samples, dtype=torch.long)

    def choose_next_node(self, current, previous, neighbors):
        weights = []
        for neighbor in neighbors:
            if neighbor == previous:
                weights.append(1 / self.p)
            elif neighbor in self.neighbors[previous]:
                weights.append(1)
            else:
                weights.append(1 / self.q)
        probabilities = torch.tensor(weights) / torch.sum(torch.tensor(weights))
        return neighbors[torch.multinomial(probabilities, 1).item()]

