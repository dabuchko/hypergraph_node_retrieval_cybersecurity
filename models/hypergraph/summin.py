import torch
from torch_geometric.nn.conv import MessagePassing
from .hyperedge_aggr import HyperedgeAggregation


class SumMin(MessagePassing):
    """
    SumMin hypergraph neural network model, supports training with mini-batches.
    """
    supports_hyperedge_attr = False
    supports_hyperedge_weight = True
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, dropout: float = 0.0):
        """
        Initializes SumMin model.
        
        :param in_channels: Number of hypernode input features.
        :type in_channels: int
        :param hidden_channels: Number of hidden channels.
        :type hidden_channels: int
        :param out_channels: Number of hypernode output features.
        :type out_channels: int
        :param num_layers: Number of layers in the model.
        :type num_layers: int
        :param dropout: Dropout rate (between 0 and 1). (default: 0.0)
        :type dropout: float
        """
        super().__init__("min", aggr_kwargs={}, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_prop = HyperedgeAggregation("sum")
        self.l1 = torch.nn.Linear(in_channels, hidden_channels)
        self.l2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.l3 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor,
                hyperedge_weight: torch.Tensor = None):
        """
        Forward pass of SumMin model.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        :param hyperedge_weight: The hyperedge weights of shape [num_edges],
        where num_edges stands for the number of hyperedges in the hypergraph.
        :type hyperedge_index: Tensor
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
        if hyperedge_weight!=None:
            hyperedge_weight = hyperedge_weight.reshape(-1, 1)
            hyperedge_weight[hyperedge_weight==0] = 1e-9

        x = self.dropout(x)
        x = self.l1(x)
        x = torch.relu(x)
        
        # propagate
        coef = 0.0
        flipped_hyperedge_index = hyperedge_index.flip([0])
        for _ in range(self.num_layers):
            coef = 2 / (3 - coef)
            hyperedge_pass = self.hyperedge_prop(x, hyperedge_index, size=(num_nodes, num_edges))
            if hyperedge_weight!=None:
                hyperedge_pass = hyperedge_pass / hyperedge_weight
            hyperedge_pass = (hyperedge_pass)
            x = (self.propagate(flipped_hyperedge_index, x=hyperedge_pass, size=(num_edges, num_nodes)) * (1-coef)) + (x*coef)
        x = self.dropout(x)
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x