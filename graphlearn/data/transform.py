from abc import ABC, abstractmethod
from typing import Union
from graphlearn.data import UnionGraph, Graph
from graphlearn.embedding import graph_embedding

class Transformer(ABC):
    """
    An abstract base class that defines a blueprint for graph transformers which can modify
    graph data structures, typically used for preprocessing steps in graph-based machine learning tasks.

    Methods:
        transform: Abstract method to transform a Graph or UnionGraph.
        fit: Method to fit the transformer based on the Graph or UnionGraph data.
        fit_transform: Method to fit and transform the Graph or UnionGraph in a single step.
    """

    @abstractmethod
    def transform(self, input: Union[Graph, UnionGraph]) -> Union[Graph, UnionGraph]:
        """
        Transforms the input Graph or UnionGraph according to the transformer's logic.

        Args:
            input (Graph | UnionGraph): The graph data structure to transform.

        Returns:
            Graph | UnionGraph: The transformed graph data structure.

        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        """
        raise NotImplementedError('Implement this method!')
    
    def fit(self, input: Union[Graph, UnionGraph]) -> 'Transformer':
        """
        Fits the transformer based on the input Graph or UnionGraph. Default implementation returns self.

        Args:
            input (Graph | UnionGraph): The graph data structure used to fit the transformer.

        Returns:
            Transformer: The instance of this transformer.
        """
        return self
    
    def fit_transform(self, input: Union[Graph, UnionGraph]) -> Union[Graph, UnionGraph]:
        """
        A convenience method that fits the transformer with the input data and then transforms the data.

        Args:
            input (Graph | UnionGraph): The graph data structure to fit and transform.

        Returns:
            Graph | UnionGraph: The transformed graph data structure.
        """
        return self.fit(input).transform(input)
    

class Node2VecEmbedding(Transformer):
    """
    A concrete transformer that applies a Node2Vec embedding to the input graph data structures.

    Attributes:
        embed_kwargs (dict): Keyword arguments necessary for the Node2Vec embedding function.
        embedding (torch.Tensor | None): Stores the embedding results post fitting.
    """

    def __init__(self, embed_kwargs) -> None:
        """
        Initializes the Node2VecEmbedding transformer with specific embedding parameters.

        Args:
            embed_kwargs (dict): Keyword arguments for the Node2Vec embedding function.
        """
        super().__init__()
        self.embed_kwargs = embed_kwargs
        self.embedding = None

    def fit(self, input: Union[Graph, UnionGraph]) -> 'Transformer':
        """
        Fits the transformer to the input Graph or UnionGraph using Node2Vec algorithm to generate embeddings.

        Args:
            input (Graph | UnionGraph): The graph data structure used to fit the transformer.

        Returns:
            Node2VecEmbedding: The instance of this transformer with generated embeddings.
        """
        if isinstance(input, Graph):
            input = input.to_union_graph()
        self.embedding = graph_embedding(input, **self.embed_kwargs)
        return self
        
    def transform(self, input: Union[Graph, UnionGraph]) -> Union[Graph, UnionGraph]:
        """
        Transforms the input Graph or UnionGraph by appending the generated Node2Vec embeddings to the node features.

        Args:
            input (Graph | UnionGraph): The graph data structure to transform.

        Returns:
            UnionGraph: The transformed graph with Node2Vec embeddings appended to node features.

        Raises:
            AssertionError: If the transformer has not been fitted with embeddings.
        """
        assert self.embedding is not None, 'First fit the transformer!'
        if isinstance(input, Graph):
            input = input.to_union_graph()
        if not isinstance(input.node_features, list):
            input.node_features = [input.node_features]
        input.node_features.append(self.embedding)
        return input
