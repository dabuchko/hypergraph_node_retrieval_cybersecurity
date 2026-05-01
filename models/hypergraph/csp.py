from torch import Tensor
from torch.nn import Module
from torch_geometric.utils import scatter

class CSP(Module):
    """
    CSP model as described in:
    CSP: An Efficient Baseline for Learning on Large-Scale Structured Data
    https://arxiv.org/pdf/2409.17628
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False
    def __init__(self, num_layers: int):
        """
        Initializes CSP model.
        
        :param num_layers: Number of layers in the model.
        :type num_layers: int
        """
        super().__init__()
        self.num_layers = num_layers

    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of CSP convolutional layer.
        
        :param x: Label matrix of shape [num_nodes, num_classes].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, num_hyperedges],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        """
        if len(x.shape)<2:
            x = x[:, None]
        # calculate the number of nodes and edges
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        
        for _ in range(self.num_layers):
            x = scatter(x[hyperedge_index[0]], hyperedge_index[1], dim=0, dim_size=num_edges, reduce="mean")
            x = scatter(x[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="mean")
        return x