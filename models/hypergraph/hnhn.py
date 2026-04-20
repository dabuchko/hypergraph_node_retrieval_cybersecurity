from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from .basic_hgnn import BasicHGNN
from torch_geometric.utils import scatter

class HNHNConv(MessagePassing):
    """
    HNHN convolution layer as described in:
    HNHN: Hypergraph Networks with Hyperedge Neurons
    https://arxiv.org/abs/2006.12278

    With slight modifications, the implementation is copied from:
    https://github.com/jianhao2016/AllSet/blob/main/src/layers.py
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 1, nonlinear_inbetween: bool = True, alpha: float = 1.0,
                 beta: float = 1.0, **kwargs):
        """
        Docstring for __init__
        
        :param self: Description
        :param in_channels: Description
        :type in_channels: int
        :param hidden_channels: Description
        :type hidden_channels: int
        :param out_channels: Description
        :type out_channels: int
        :param heads: Description
        :type heads: int
        :param nonlinear_inbetween: Description
        :type nonlinear_inbetween: bool
        :param alpha: Description
        :type alpha: float
        :param beta: Description
        :type beta: float
        :param kwargs: Description
        """
        super(HNHNConv, self).__init__("add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween

        # preserve variable heads for later use (attention)
        self.heads = heads
        self.concat = True
        # self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)
        self.alpha = alpha
        self.beta = beta

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()
        # glorot(self.weight_v2e)
        # glorot(self.weight_e2v)
        # zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor):
        """
        Forward pass of HNHN convolutional layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
            


        # the degree of the node
        DV = scatter(x.new_ones([hyperedge_index.size(1)]), hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        # the degree of the hyperedge
        DE = scatter(x.new_ones([hyperedge_index.size(1)]), hyperedge_index[1],
                     dim=0, dim_size=num_edges, reduce='sum')

        # alpha part
        D_e_alpha = DE ** self.alpha
        D_v_alpha = scatter(DE[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="sum")

        # beta part
        D_v_beta = DV ** self.beta
        D_e_beta = scatter(DV[hyperedge_index[0]], hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")

        D_v_alpha_inv = 1.0 / D_v_alpha
        D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

        D_e_beta_inv = 1.0 / D_e_beta
        D_e_beta_inv[D_e_beta_inv == float("inf")] = 0


        x = self.weight_v2e(x)

#         ipdb.set_trace()
#         x = torch.matmul(torch.diag(data.D_v_beta), x)
        x = D_v_beta.unsqueeze(-1) * x

        out = self.propagate(hyperedge_index, x=x, norm=D_e_beta_inv,
                             size=(num_nodes, num_edges))
        
        if self.nonlinear_inbetween:
            out = F.relu(out)
        
        out = self.weight_e2v(out)
        
#         out = torch.matmul(torch.diag(data.D_e_alpha), out)
        out = D_e_alpha.unsqueeze(-1) * out

        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D_v_alpha_inv,
                             size=(num_edges, num_nodes))
        
        return out

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.hidden_channels, self.out_channels)


class HNHN(BasicHGNN):
    """
    HNHN model as described in:
    HNHN: Hypergraph Networks with Hyperedge Neurons
    https://arxiv.org/abs/2006.12278
    """
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False
    def init_conv(self, in_channels: int, out_channels: int, heads: int = 1, nonlinear_inbetween=True, alpha=1.0, beta=1.0):
        return HNHNConv(in_channels, out_channels, out_channels, heads, nonlinear_inbetween, alpha, beta)