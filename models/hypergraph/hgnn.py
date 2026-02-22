from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import HypergraphConv

class HGNN(BasicHGNN):
    supports_hyperedge_weight = True
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, dropout: float = 0.0, bias: bool = False):
        return HypergraphConv(in_channels, out_channels, False, dropout=dropout, bias=bias)