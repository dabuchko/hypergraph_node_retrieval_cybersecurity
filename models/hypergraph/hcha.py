from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import HypergraphConv

class HCHA(BasicHGNN):
    supports_hyperedge_weight = True
    supports_hyperedge_attr = True
    def __init__(self, in_channels, num_layers, out_channels = None, dropout = 0, act = "relu", act_first = False, act_kwargs = None, norm = None, norm_kwargs = None, jk = None, **kwargs):
        super().__init__(in_channels, in_channels, num_layers, out_channels, dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

    def init_conv(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0, bias: bool = False):
        return HypergraphConv(in_channels, out_channels, True, concat=False, heads=heads, dropout=dropout, bias=bias)