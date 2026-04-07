from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import HypergraphConv

class HCHA(BasicHGNN):
    """
    Hypergraph Convolution and Hypergraph Attention (HCHA) model as described in:
    Hypergraph Convolution and Hypergraph Attention
    https://arxiv.org/abs/1901.08150
    """
    supports_hyperedge_weight = True
    supports_hyperedge_attr = True
    def __init__(self, in_channels, num_layers, out_channels = None, dropout = 0, act = "relu", act_first = False, act_kwargs = None, norm = None, norm_kwargs = None, jk = None, **kwargs):
        """
        Initializes HCHA model.
        
        Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
        """
        super().__init__(in_channels, in_channels, num_layers, out_channels, dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

    def init_conv(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0, bias: bool = False):
        return HypergraphConv(in_channels, out_channels, True, concat=False, heads=heads, dropout=dropout, bias=bias)