from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from typing import (List, Optional, Union)
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Size

class HyperedgeAggregation(MessagePassing):
    """
    Aggregation layer for hyperedges. It aggregates node features to hyperedge features based on the provided aggregation method.
    """
    def __init__(self, aggr: Optional[Union[str, List[str], Aggregation]] = "mean", **kwargs):
        """
        Initializes the hyperedge aggregation layer.
        
        :param aggr: The aggregation scheme
            to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"` or :obj:`"mul"`.
            In addition, can be any
            :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
            that automatically resolves to it).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
            If set to :obj:`None`, the :class:`MessagePassing` instantiation is
            expected to implement its own aggregation logic via
            :meth:`aggregate`. (default: :obj:`"add"`)
        :type aggr: Optional[Union[str, List[str], Aggregation]]
        :param kwargs: Additional arguments to be passed to torch_geometric.nn.conv.MessagePassing class.
        """
        super().__init__(aggr, **kwargs)
    def forward(self, x: Tensor, hyperedge_index: Tensor, size: Size = None):
        """
        Forward pass of hyperedge aggregation layer.
        
        :param x: Feature matrix of shape [num_nodes, num_features].
        :type x: Tensor
        :param hyperedge_index: The hyperedge indices of shape [2, K],
        where the first row contains hypernode indices and the second -- hyperedge indices.
        :type hyperedge_index: Tensor
        :param size: A tuple (num_nodes, num_edges) where num_nodes is the number of hypernodes,
        and num_edges is the number of hyperedges.
        :type size: Size
        """
        return self.propagate(hyperedge_index, x=x, size=size)