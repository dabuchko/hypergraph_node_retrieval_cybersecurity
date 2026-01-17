from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import HypergraphConv

class HCHA(BasicHGNN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0):
        return HypergraphConv(in_channels, out_channels, True, concat=False, heads=heads, dropout=dropout)