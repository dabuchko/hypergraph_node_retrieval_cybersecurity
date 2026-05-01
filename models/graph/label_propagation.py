from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter

class LabelPropagation(Module):
    """
    Label Propagation model as described in:
    Learning from Labeled and Unlabeled Data with Label Propagation
    http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf

    Compared to PyTorch Geometric implementation of Label Propagation,
    this implementation is simpler and more memory efficient
    """
    def __init__(self, num_layers: int, alpha: float):
        """
        Initializes Label Propagation model.
        
        :param num_layers: Number of layers in the model.
        :type num_layers: int
        :param alpha: The coefficient alpha is multiplier for the aggregated labels,
        and (1-alpha) is multiplier for the initial labels.
        :type alpha: float
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None):
        """
        Forward pass of Label Propagation convolutional layer.
        
        :param x: Label matrix of shape [num_nodes, num_classes].
        :type x: Tensor
        :param edge_index: The directed edge indices of shape [2, num_edges].
        :type edge_index: Tensor
        :param edge_weight: The edge weights, ignored in this implementation. (default: None)
        :type edge_weight: Tensor
        """
        if len(x.shape)<2:
            x = x[:, None]
        num_nodes = x.size(0)
        
        for _ in range(self.num_layers):
            out = scatter(x[edge_index[0]], edge_index[1], dim=0, dim_size=num_nodes, reduce="mean")
            x = (self.alpha * out) + ((1 - self.alpha) * x)
        return x