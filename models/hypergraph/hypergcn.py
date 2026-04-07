import torch
import torch_geometric
from torch.nn import Module
from torch import Tensor, ones
from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

class HyperGCNConv(MessagePassing):
    """
    HyperGCN convolution as described in:
    HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs
    https://arxiv.org/abs/1809.02589
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        """
        Initializes HyperGCN convolutional layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param bias: Whether to use bias in the linear transformation (default False).
        :type bias: bool
        """
        super().__init__("sum", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of HyperGCN convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        """
        x = self.lin(x)
        num_nodes = x.shape[0]
        with torch.no_grad():
            edge_index = []
            edge_weight = []
            hyperedge_dict = {} # key is hyperedge index, values are hypernodes indecies
            for i in range(hyperedge_index.shape[1]):
                node = hyperedge_index[0, i].item()
                edge = hyperedge_index[1, i].item()
                if edge in hyperedge_dict:
                    hyperedge_dict[edge].append(node)
                else:
                    hyperedge_dict[edge] = [node]
            for i in range(num_nodes):
                edge_index.append((i,i))
                edge_weight.append(1.0)
            for hyperedge in hyperedge_dict.keys():
                if len(hyperedge_dict[hyperedge])<=1:
                    continue
                max_dist = -1
                best_i, best_j = -1, -1
                for i in range(len(hyperedge_dict[hyperedge]) - 1):
                    a = hyperedge_dict[hyperedge][i]
                    b = hyperedge_dict[hyperedge][i+1:]
                    d = ((x[b] - x[a][None, :])**2).sum(1)
                    max_d = d.max().item()
                    if max_dist<max_d:
                        max_dist = max_d
                        best_i = a
                        best_j = b[d.argmax().item()]
                w = 1/(2*len(hyperedge_dict[hyperedge]) - 3)
                edge_index.append((best_j, best_i))
                edge_index.append((best_i, best_j))
                edge_weight.append(w)
                edge_weight.append(w)
                for n in hyperedge_dict[hyperedge]:
                    if n!=best_i and n!=best_j:
                        edge_index.append((n, best_i))
                        edge_index.append((n, best_j))
                        edge_index.append((best_i, n))
                        edge_index.append((best_j, n))
                        for _ in range(4):
                            edge_weight.append(w)
            edge_index = torch.tensor(edge_index, device=x.device).T
            edge_weight = torch.tensor(edge_weight, device=x.device)
        return self.propagate(edge_index, x=x, size=(num_nodes, num_nodes), edge_weight=edge_weight)


class HyperGCN(BasicHGNN):
    """
    HyperGCN model as described in:
    HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs
    https://arxiv.org/abs/1809.02589
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, bias: bool = False, **kwargs):
        return HyperGCNConv(in_channels, out_channels, bias)