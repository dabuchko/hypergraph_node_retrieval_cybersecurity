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
        
    def forward(self, x: Tensor, hyperedge_index: Tensor):
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
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        out = self.hyperedge_aggr(x, hyperedge_index, (num_nodes, num_edges))
        out = self.bn(out)
        out = self.relu(out)
        out = self.propagate(hyperedge_index.flip([0]), x=out, size=(num_edges, num_nodes))
        out.add_(x)
        return self.lin(out)

class SumMin(BasicHGNN):
    """
    SumMin hypergraph model.
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: int = None, dropout: float = 0.0, num_nodes: int = None,
                 **kwargs):
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
        :param num_nodes: The number of hypernodes in the hypergraph. If not provided,
        deduced from hyperedge_index at the first forward pass. (default: None)
        :type num_nodes: int
        :param kwargs: The remaining parameters to be passed to the parent BasicHGNN class.
        """
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, None, **kwargs)
        if num_nodes is not None:
            self.x = torch.nn.Parameter(torch.empty((num_nodes, in_channels)))
            torch.nn.init.xavier_uniform_(self.x)
        else:
            self.x = torch.nn.UninitializedParameter()
    def forward(self, hyperedge_index: Tensor, batch_hypernodes: Tensor = None):
        """
        Forward pass of MinSum model.
        
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        :param batch_hypernodes: This parameter should be used during training with mini-batches.
        It is a tensor of size (N, ), where position i has value j if jth feature vector
        should be used in the current forward pass under index i.
        :type batch_hypernodes: Tensor
        """
        if hyperedge_index is None:
            raise Exception("'hyperedge_index' argument cannot be None.")
        if isinstance(self.x, torch.nn.UninitializedParameter):
            self.x.materialize((hyperedge_index[0].max().item()+1, self.in_channels), device=hyperedge_index.device)
            torch.nn.init.xavier_uniform_(self.x.data)
        if batch_hypernodes is None:
            x = self.x
        else:
            x = self.x[batch_hypernodes]
        return super().forward(x, hyperedge_index)

    def init_conv(self, in_channels: int, out_channels: int):
        return SumMinConv(in_channels, out_channels)


class SumMinAblation(BasicHGNN):
    """
    Ablation of SumMin hypergraph model, where instead of trainable embedding
    the external embedding is used and additional linear layer with tanh layer
    is applied to it before SumMin convolutions.
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: int = None, dropout: float = 0.0, **kwargs):
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
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, dropout, None, **kwargs)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(in_channels, in_channels)
        )
        
    def forward(self, x: Tensor = None, hyperedge_index: Tensor = None):
        """
        Forward pass of MinSum model.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        """
        return super().forward(self.mlp(x), hyperedge_index)

    def init_conv(self, in_channels: int, out_channels: int):
        return SumMinConv(in_channels, out_channels)
    