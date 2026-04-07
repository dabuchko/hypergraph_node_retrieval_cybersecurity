import torch
from torch import Tensor, ones
from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter
from .hyperedge_aggr import HyperedgeAggregation


class UniGCNConv(MessagePassing):
    """
    UniGCN hypergraph convolution layer as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False):
        """
        Initializes the UniGCN convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param hyperedge_aggr: The aggregation scheme to use for hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        :param bias: Whether to use bias in the linear transformation. (default: False)
        :type bias: bool
        """
        super().__init__(hyperedge_aggr, aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.hyperedge_aggr = hyperedge_aggr
        assert hyperedge_aggr=="mean" or hyperedge_aggr=="sum"
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of UniGCN convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
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

        D = scatter(ones([hyperedge_index.size(1)], device=hyperedge_index.device), hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')

        B = scatter(D[hyperedge_index[0]], hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='mean')
        D = 1.0 / torch.sqrt(D+1)
        D[D == float("inf")] = 0
        B = 1.0 / torch.sqrt(B)
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, size=(num_nodes, num_edges))
        out = self.lin(out)
        out *= B[:, None]
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        if self.hyperedge_aggr=="mean":
            out /= D[:, None]
            out += self.lin(x) * (D**2)[:, None] # self loops
        elif self.hyperedge_aggr=="sum":
            out += self.lin(x) * D[:, None] # self loops
            out *= D[:, None]
        return out

class UniGCN(BasicHGNN):
    """
    UniGCN hypergraph model as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False, **kwargs):
        return UniGCNConv(in_channels, out_channels, hyperedge_aggr, bias)

class UniGATConv(MessagePassing):
    """
    UniGAT hypergraph convolution layer as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False, negative_slope: float = 0.01):
        """
        Initializes the UniGAT convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param hyperedge_aggr: The aggregation scheme to use for hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        :param bias: Whether to use bias in the linear transformation. (default: False)
        :type bias: bool
        :param negative_slope: Negative slope for the LeakyReLU activation used in attention mechanism. (default: 0.01)
        :type negative_slope: float
        """
        super().__init__(hyperedge_aggr, aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.a = torch.nn.Parameter(torch.empty((out_channels*2,)))
        torch.nn.init.uniform_(self.a)
        self.negative_slope = negative_slope
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of UniGAT convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
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
        x = self.lin(x)
        out = self.propagate(hyperedge_index, x=x, size=(num_nodes, num_edges))
        x_att = (x * self.a[:out.shape[1]][None, :]).sum(1)
        out_att = (out * self.a[out.shape[1]:][None, :]).sum(1)
        att = torch.nn.functional.leaky_relu(x_att[hyperedge_index[0]] + out_att[hyperedge_index[1]], self.negative_slope)
        att_self = torch.nn.functional.leaky_relu(x_att*2, self.negative_slope)
        att = att - max(att.max(), att_self.max())
        att_self = att_self - max(att.max(), att_self.max())
        att = torch.exp(att)
        att_self = torch.exp(att_self)
        att_sum = scatter(att, hyperedge_index[0], dim=0, dim_size=num_nodes, reduce='sum') + att_self + 1e-9
        att = att / att_sum[hyperedge_index[0]]
        att_self = att_self / att_sum
        res = att_self[:, None] * x
        step = 3000
        for i in range(0, hyperedge_index.shape[1], step):
            res += scatter(att[i:i+step][:, None] * out[hyperedge_index[1,i:i+step]], hyperedge_index[0, i:i+step], dim=0, dim_size=num_nodes, reduce='sum')
        if (res!=res).sum().item()>0:
            breakpoint()
        return res

class UniGAT(BasicHGNN):
    """
    UniGAT hypergraph model as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False, **kwargs):
        return UniGATConv(in_channels, out_channels, hyperedge_aggr, bias)

class UniGINConv(MessagePassing):
    """
    UniGIN hypergraph convolution layer as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False):
        """
        Initializes the UniGIN convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param hyperedge_aggr: The aggregation scheme to use for hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        :param bias: Whether to use bias in the linear transformation. (default: False)
        :type bias: bool
        """
        super().__init__("sum", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_aggr = HyperedgeAggregation(hyperedge_aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.eps = torch.nn.Parameter(torch.rand(1))
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of UniGIN convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
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

        out = self.hyperedge_aggr(x, hyperedge_index, (num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out += x * (1 + self.eps)
        out = self.lin(out)
        return out

class UniGIN(BasicHGNN):
    """
    UniGIN hypergraph model as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, hyperedge_aggr="mean", bias: bool = False, **kwargs):
        return UniGINConv(in_channels, out_channels, hyperedge_aggr, bias)

class UniSAGEConv(MessagePassing):
    """
    UniSAGE hypergraph convolution layer as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_channels: int, out_channels: int, sage_aggr="max", hyperedge_aggr="mean", bias: bool = False):
        """
        Initializes the UniSAGE convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param sage_aggr: The aggregation scheme to use for sage aggregation of hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        :param hyperedge_aggr: The aggregation scheme to use for hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        :param bias: Whether to use bias in the linear transformation. (default: False)
        :type bias: bool
        """
        super().__init__(sage_aggr, aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_aggr = HyperedgeAggregation(hyperedge_aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of UniSAGE convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
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

        out = self.hyperedge_aggr(x, hyperedge_index, (num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out += x
        out = self.lin(out)
        return out

class UniSAGE(BasicHGNN):
    """
    UniSAGE hypergraph model as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, sage_aggr="max", hyperedge_aggr="mean", bias: bool = False):
        return UniSAGEConv(in_channels, out_channels, sage_aggr, hyperedge_aggr, bias)

class UniGCNIIConv(MessagePassing):
    """
    UniGCNII hypergraph convolution layer as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.0, beta: float = 1.0, hyperedge_aggr="mean"):
        """
        Initializes the UniGCNII convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param alpha: Portion between 0.0 and 1.0 of previous layer output to include and current output to exclude (default: 0.0)
        :type alpha: float
        :param beta: Portion between 0.0 and 1.0 of weight matrix to include in linear transformation and portion of identity matrix to exclude (default: 1.0)
        :type beta: float
        :param hyperedge_aggr: The aggregation scheme to use for hyperedges (commonly "sum" or "mean").
        All available aggregation methods are described in torch_geometric.nn.aggr.
        """
        super().__init__("sum", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self._W = torch.nn.Parameter(torch.empty((in_channels, out_channels)))
        torch.nn.init.xavier_uniform_(self._W)
        self.hyperedge_aggr = HyperedgeAggregation(hyperedge_aggr)
        assert hyperedge_aggr=="mean" or hyperedge_aggr=="sum"
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of UniGCNII convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
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

        D = scatter(ones([hyperedge_index.size(1)], device=hyperedge_index.device), hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')

        B = scatter(D[hyperedge_index[0]], hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='mean')
        D = 1.0 / torch.sqrt(D+1)
        D[D == float("inf")] = 0
        B = 1.0 / torch.sqrt(B)
        B[B == float("inf")] = 0

        out = self.hyperedge_aggr(x, hyperedge_index, (num_nodes, num_edges))
        out *= B[:, None]
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out *= D[:, None]
        out = ((1 - self.alpha) * out) + (self.alpha * x)
        weight = (self.beta*self._W) + ((1-self.beta) * torch.eye(self._W.shape[0], self._W.shape[1], device=self._W.device))
        return out @ weight

class UniGCNII(BasicHGNN):
    """
    UniGCNII hypergraph model as described in:
    UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks
    https://arxiv.org/abs/2105.00956
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, alpha: float = 0.0, beta: float = 1.0, hyperedge_aggr="mean", **kwargs):
        return UniGCNIIConv(in_channels, out_channels, alpha, beta, hyperedge_aggr)