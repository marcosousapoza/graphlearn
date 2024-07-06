import torch
import torch_scatter
from torch import nn


class ScatterWrapper(nn.Module):
    """
    A wrapper class for the torch_scatter.scatter function to provide aggregation operations.

    :param aggregation: The aggregation method to use ('sum', 'mean', 'max', 'min', 'mul', 'div', 'add', 'sub', etc.).
    :type aggregation: str
    """
    def __init__(self, aggregation='mean'):
        super(ScatterWrapper, self).__init__()
        self.aggregation = aggregation


    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass for the scatter operation.

        :param batch: A tuple containing the tensor to scatter and the index tensor.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :return: The result of the scatter operation.
        :rtype: torch.Tensor
        """
        y, idx = batch
        if self.aggregation == 'std':
            y += torch.ones_like(y) * 1e-4
            z = torch_scatter.composite.scatter_std(y, idx, dim=0)
        else:
            z = torch_scatter.scatter(y, idx, dim=0, reduce=self.aggregation)
        return z