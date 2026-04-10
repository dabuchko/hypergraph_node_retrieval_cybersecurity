import torch
from torch_geometric.nn.conv import MessagePassing
from .hyperedge_aggr import HyperedgeAggregation
from torch import Tensor
from .basic_hgnn import BasicHGNN

class SumMinConv(MessagePassing):
    """
    SumMin hypergraph neural network convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the SumMin convolution layer.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        """
        super().__init__("min", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_aggr = HyperedgeAggregation("sum")
        self.bn = torch.nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.bn.bias.data.fill_(1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x: Tensor, hyperedge_index: Tensor, hyperedge_weight: Tensor = None):
        """
        Forward pass of MinSum convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        :param hyperedge_weight: The hyperedge weights of shape [num_edges],
        where num_edges stands for the number of hyperedges in the hypergraph.
        None if no hyperedge weights are provided. (default: None)
        :type hyperedge_weight: Tensor
        """
        if len(x.shape)<2:
            x = x[:, None]
        # calculate the number of nodes and edges
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_weight is not None:
            num_edges = hyperedge_weight.shape[0]
        elif hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        out = self.hyperedge_aggr(x, hyperedge_index, (num_nodes, num_edges))
        out = self.bn(out)
        out = self.relu(out)
        if hyperedge_weight is not None:
            out = out / (hyperedge_weight.reshape(-1, 1) / (hyperedge_weight.mean()+1e-9) + 1e-9)
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out.add_(x)
        return self.lin(out)

class SumMin(BasicHGNN):
    """
    SumMin hypergraph model.
    """
    supports_hyperedge_weight = True
    supports_hyperedge_attr = False

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int = None,
                 dropout: float = 0.0, **kwargs):
        """
        Initializes the SumMin model.
        
        :param in_channels: Number of input features.
        :type in_channels: int
        :param hidden_channels: Number of hidden channels.
        :type hidden_channels: int
        :param num_layers: Number of convolutional layers in the model.
        :type num_layers: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param dropout: The dropout rate that is applied between convolutional layers. (default: 0.0)
        :type dropout: float
        :param kwargs: The remaining parameters to be passed to the parent BasicHGNN class.
        """
        super().__init__(hidden_channels, hidden_channels, num_layers, out_channels, dropout, None, **kwargs)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.Tanh()
        )
    def forward(self, x: Tensor, hyperedge_index: Tensor, hyperedge_weight: Tensor = None):
        """
        Forward pass of MinSum model.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        :param hyperedge_weight: The hyperedge weights of shape [num_edges],
        where num_edges stands for the number of hyperedges in the hypergraph.
        None if no hyperedge weights are provided. (default: None)
        :type hyperedge_weight: Tensor
        """
        if len(x.shape)<2:
            x = x[:, None]
        x = self.mlp(x)
        return super().forward(x, hyperedge_index, hyperedge_weight)

    def init_conv(self, in_channels: int, out_channels: int):
        return SumMinConv(in_channels, out_channels)