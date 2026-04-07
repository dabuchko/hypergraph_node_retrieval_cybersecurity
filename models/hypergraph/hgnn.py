from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import HypergraphConv

class HGNN(BasicHGNN):
    """
    Hypergraph Neural Network (HGNN) as described in:
    Hypergraph Neural Networks
    https://arxiv.org/abs/1809.09401

    Since PyTorch Geometric's implementation of HypergraphConv without attention
    is similar to HGNN convolutional layer, it is used instead, with attention_mode set to False. 
    """
    supports_hyperedge_weight = True
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, dropout: float = 0.0, bias: bool = False):
        return HypergraphConv(in_channels, out_channels, False, dropout=dropout, bias=bias)