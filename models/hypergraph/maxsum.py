import torch
from torch_geometric.nn.conv import MessagePassing
from .unignn import HyperedgeAggregation

class MaxSum(MessagePassing):
    supports_hyperedge_attr = False
    supports_hyperedge_weight = False
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__("sum", aggr_kwargs={}, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_prop = HyperedgeAggregation("max")
        self.l1 = torch.nn.Linear(in_channels, hidden_channels)
        self.l3 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x, hyperedge_index):
        if len(x.shape)<2:
            x = x[:, None]
        # calculate the number of nodes and edges
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        x = self.dropout(x)
        x = -torch.relu(self.l1(x))
        
        # propagate
        coef = 0.0
        flipped_hyperedge_index = hyperedge_index.flip([0])
        for _ in range(self.num_layers):
            coef = 2 / (3 - coef)
            hyperedge_pass = self.hyperedge_prop(x, hyperedge_index, size=(num_nodes, num_edges))
            x = (self.propagate(flipped_hyperedge_index, x=hyperedge_pass, size=(num_edges, num_nodes)) * (1-coef)) + (x*coef)
        
        x = self.dropout(x)
        x = self.l3(x)
        return x