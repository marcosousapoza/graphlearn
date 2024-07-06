from torch.utils.data import Dataset
import graphlearn.data as data
from typing import List

class MultiGraphDataset(Dataset):
    """
    Dataset class for handling a collection of graphs.
    """
    def __init__(self, graphs: List[data.Graph]):
        """
        Initializes the dataset with a list of Graph instances.
        
        Args:
            graphs (List[data.Graph]): A list of Graph instances.
        """
        self.graphs = graphs

    def __len__(self) -> int:
        """
        Return the number of graphs in the dataset.
        
        Returns:
            int: The number of graphs.
        """
        return len(self.graphs)

    def __getitem__(self, idx: int) -> data.Graph:
        """
        Fetches the Graph at the provided index.
        
        Args:
            idx (int): Index of the graph to retrieve.
        
        Returns:
            data.Graph: A single Graph instance.
        """
        return self.graphs[idx]


class SingleGraphDataset(Dataset):
    """
    Dataset class for handling a single graph.
    """
    def __init__(self, graph: data.Graph):
        """
        Initializes the dataset with a single Graph instance.
        
        Args:
            graph (data.Graph): A Graph instance.
        """
        self.graph = graph

    def __len__(self) -> int:
        """
        Return the number of graphs in the dataset.
        
        Returns:
            int: The number of graphs.
        """
        return 1

    def __getitem__(self, idx: int) -> List[data.Graph]:
        """
        Fetches the single Graph as a singleton list.
        
        Args:
            idx (int): Index of the graph to retrieve (redundant).
        
        Returns:
            data.Graph: A single Graph instance.
        """
        return [self.graph]
