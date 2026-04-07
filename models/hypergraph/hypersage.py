import torch
import torch_geometric
from .basic_hgnn import BasicHGNN
from torch_geometric.utils import scatter


class HyperSAGEConv(torch_geometric.nn.conv.MessagePassing):
    """
    HyperSAGE convolution as described in:
    HyperSAGE: Generalizing Inductive Representation Learning on Hypergraphs
    https://arxiv.org/abs/2010.04558
    """
    def __init__(self, in_channels: int, out_channels: int, p: float = 1.0):
        """
        Initializes HyperSAGE convolutional layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param p: Power to which the features are raised during aggregation (default 1.0).
        :type p: float
        """
        super().__init__("sum", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.p = p
        self.lin = torch.nn.Linear(in_channels, out_channels)
    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor):
        """
        Forward pass of HNHN convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        """
        # calculate the number of nodes and edges
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        D = scatter(x.new_ones([hyperedge_index.size(1)]), hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        # NOTE: B may include duplicates, the original implementation suggests
        # to compute the number of unique hypernode neighbors accross all incident
        # hyperedges, however this is too computationally inefficient.
        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = scatter(B[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0
        
        out = self.propagate(hyperedge_index, x=torch.sgn(x) * ((torch.abs(x)+1e-9) ** self.p), size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out = out * (D * B).unsqueeze(1)
        out = torch.sgn(out) * ((torch.abs(out)+1e-9)  ** (1/self.p))
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-9)
        out = x + out
        return self.lin(out)

class HyperSAGE(BasicHGNN):
    """
    HyperSAGE model as described in:
    HyperSAGE: Generalizing Inductive Representation Learning on Hypergraphs
    https://arxiv.org/abs/2010.04558
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False
    def init_conv(self, in_channels: int, out_channels: int, p: float = 1.0):
        return HyperSAGEConv(in_channels, out_channels, p)