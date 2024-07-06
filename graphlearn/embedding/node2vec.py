from typing import Union
import torch
import torch.nn as nn
from graphlearn.data import UnionGraph, PQWalkSampler, Graph
from torch.utils.data import DataLoader


def graph_embedding(
        union_graph: Union[UnionGraph, Graph], embedding_dim: int, walk_length: int, neg_samples_length: int, 
        p: float = 1.0, q: float = 1.0, batch_size = 100, learning_rate: float = 0.01,
        max_iter: int = 1000, patience: int = 30
    ):
    """
    Embeds a UnionGraph using the Node2Vec model, training the model on walks generated from the graph.
    The function includes a patience mechanism to stop training early if loss improvement stalls.

    Args:
        union_graph (UnionGraph): The union graph to embed, containing all graphs' data.
        embedding_dim (int): Dimensionality of the embeddings to be learned.
        walk_length (int): Length of each random walk.
        neg_samples_length (int): Number of negative samples for each walk.
        p (float): Node2Vec return hyperparameter (default: 1.0).
        q (float): Node2Vec in-out hyperparameter (default: 1.0).
        batch_size (int): Number of walks processed per batch.
        learning_rate (float): Learning rate for the optimizer.
        max_iter (int): Maximum number of iterations for training.
        patience (int): Number of iterations to wait for improvement before stopping early.

    Returns:
        torch.Tensor: The embedding matrix for all nodes in the union graph.
    """
    # Instantiate the Node2Vec model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_nodes = union_graph.number_of_nodes()
    model = Node2Vec(total_nodes, embedding_dim)
    model.to(device=device)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the sampler
    sampler = PQWalkSampler(union_graph, p, q, walk_length, neg_samples_length)
    data_loader = DataLoader(sampler, batch_size=batch_size, num_workers=5)  # Single batch; adjust num_workers as needed

    # Training loop
    max_loss, pat = float('inf'), 100
    for i, (walk, neg_walk) in enumerate(data_loader):
        # Move tensors to correct device
        walk, neg_walk = walk.to(device), neg_walk.to(device)
        optimizer.zero_grad()
        loss = model(walk, neg_walk)  # Dummy walk input since we're only generating starts and negative samples here
        loss.backward()
        optimizer.step()

        if loss.item() + 1e-3 < max_loss:
            max_loss = loss.item()
            pat = patience
        elif pat > 0:
            pat -= 1
        else:
            break

        if i == max_iter - 1:
            raise RuntimeError('Function did not converge. Consider increasing `max_iter`')

    # Retrieve and return the final node embeddings
    return model.get_embeddings()


class Node2Vec(nn.Module):
    def __init__(self, nodes: int, embedding_dim: int) -> None:
        """
        Initialize the Node2Vec model with an embedding layer.
        Args:
            nodes (int): Total number of nodes across all graphs.
            embedding_dim (int): Dimension of the embeddings.
        """
        super(Node2Vec, self).__init__()
        self.embedding = nn.Embedding(nodes, embedding_dim)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, walk: torch.Tensor, neg_walk: torch.Tensor):
        """
        Forward pass of Node2Vec, calculating the loss for node embeddings
        while considering graph boundaries specified by graph_idx.
        Args:
            walk (torch.Tensor): Nodes in each walk (batch_size, walk_length).
            neg_walk (torch.Tensor): Negative samples for each walk (batch_size, neg_sample_size).
            graph_idx (torch.Tensor): Index indicating which graph each walk belongs to.
        Returns:
            torch.Tensor: Total loss for the node embeddings.
        """
        start_embed = self.embedding(walk[:, 0])
        walk_embed = self.embedding(walk[:, 1:])
        neg_walk_embed = self.embedding(neg_walk)

        
        # Concatenate positive and negative walks
        combined_embed = torch.cat((walk_embed, neg_walk_embed), dim=1)  # (batch_size, walk_length + neg_sample_size, embedding_dim)
        # Calculate positive scores
        pos_score = torch.sum(start_embed.unsqueeze(1) * walk_embed, dim=2)  # (batch_size, walk_length)
        # Calculate combined scores for normalization
        combined_score = torch.sum(start_embed.unsqueeze(1) * combined_embed, dim=2)  # (batch_size, walk_length + neg_sample_size)
        # Compute loss
        dividend = -torch.sum(pos_score, dim=1)  # Sum over the walk_length
        quotient = walk.size(1) * torch.log(torch.sum(torch.exp(combined_score), dim=1))  # Sum over walk_length + neg_sample_size
        loss = dividend + quotient

        # Then sum the mean losses of all graphs
        total_loss = torch.sum(loss)

        return total_loss

    def get_embeddings(self) -> torch.Tensor:
        """
        Retrieves the learned node embeddings.
        Returns:
            torch.Tensor: The node embeddings tensor.
        """
        return self.embedding.weight.detach()