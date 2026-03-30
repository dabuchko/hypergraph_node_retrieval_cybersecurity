import torch
from torch_geometric.nn.conv import MessagePassing
from .unignn import HyperedgeAggregation

class MinSum(MessagePassing):
    supports_hyperedge_attr = False
    supports_hyperedge_weight = True
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__("sum", aggr_kwargs={}, flow="source_to_target", node_dim=-2, decomposed_layers=1)
        self.hyperedge_prop = HyperedgeAggregation("min")
        self.l1 = torch.nn.Linear(in_channels, hidden_channels)
        self.l2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout, True)
        self.num_layers = num_layers

    def forward(self, x, hyperedge_index, hyperedge_weight = None):
        if len(x.shape)<2:
            x = x[:, None]
        # calculate the number of nodes and edges
        num_nodes = x.size(0)
        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight!=None:
            hyperedge_weight = hyperedge_weight.reshape(-1, 1)
            hyperedge_weight[hyperedge_weight==0] = 1e-9

        x = self.dropout(x)
        x = torch.relu(self.l1(x))
        
        # propagate
        coef = 0.0
        flipped_hyperedge_index = hyperedge_index.flip([0])
        for _ in range(self.num_layers):
            coef = 2 / (3 - coef)
            hyperedge_pass = self.hyperedge_prop(x, hyperedge_index, size=(num_nodes, num_edges))
            if hyperedge_weight!=None:
                hyperedge_pass = hyperedge_pass / hyperedge_weight
            x = (self.propagate(flipped_hyperedge_index, x=hyperedge_pass, size=(num_edges, num_nodes)) * (1-coef)) + (x*coef)
        
        x = self.dropout(x)
        x = self.l2(x)
        return x