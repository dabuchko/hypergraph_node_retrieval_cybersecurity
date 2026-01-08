from torch import Tensor, ones
from .basic_hgnn import BasicHGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter

class CSPConv(MessagePassing):
    def __init__(self):
        super().__init__("sum", aggr_kwargs=None, flow="source_to_target", node_dim=-2, decomposed_layers=1)
    def forward(self, x: Tensor, hyperedge_index: Tensor):
        # calculate the number of nodes and edges
        num_nodes = hyperedge_index[0].max() + 1
        num_edges = hyperedge_index[1].max() + 1
        D = scatter(ones([hyperedge_index[1]]), hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=B.reshape(-1, 1)*x,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=D.reshape(-1, 1)*out, size=(num_edges, num_nodes))

class CSP(BasicHGNN):
    supports_hyperedge_weight = False
    supports_hyperedge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return CSPConv(in_channels, out_channels, **kwargs)