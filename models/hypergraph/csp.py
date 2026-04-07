from torch import Tensor, ones
from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

class CSPConv(MessagePassing):
    """
    CSP convolutional layer as described in:
    CSP: An Efficient Baseline for Learning on Large-Scale Structured Data
    https://arxiv.org/pdf/2409.17628
    """
    def __init__(self):
        super().__init__("mean", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of CSP convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
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

        out = self.propagate(hyperedge_index, x=x, size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        return out

class CSP(BasicHGNN):
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
        super().__init__(1, 1, num_layers, 1)

    def init_conv(self, _: int, __: int, **kwargs):
        return CSPConv()