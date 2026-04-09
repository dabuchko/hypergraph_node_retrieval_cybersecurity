import torch
from data import Hypergraph
from models.hypergraph.hyperedge_aggr import HyperedgeAggregation

def hyper_undersampling(hypergraph: Hypergraph, l: int = 3, alpha: float = 0.1):
    """
    Performs undersampling of the majority class in the hypergraph, according to:
    Hypergraph-based importance assessment for binary classification data
    https://link.springer.com/article/10.1007/s10115-022-01786-2
    Except that instead of random walks that approximate node rating, the
    standard mean aggregation and propagation methods are used.
    
    :param hypergraph: Hypergraph to be undersampled.
    :type hypergraph: Hypergraph
    :param l: Number of iterations (aggregation and propagation steps). Equivalent
    to the length of random walks in the original implementation.
    :type l: int
    :param alpha: Randomness factor. The multiplier of random noise added to the node rate.
    :type alpha: float
    """
    # compute node importance scores by aggregation and propagation methods
    mean_aggr = HyperedgeAggregation()
    y = hypergraph.y.float().reshape(-1, 1)
    y[~hypergraph.train_mask] = 0 # ignore non-training labels
    positive_num = int(y.sum().item())
    for _ in range(l):
        y = mean_aggr.propagate(hypergraph.hyperedge_index, x=y, size=(hypergraph.num_nodes, hypergraph.num_edges))
        y = mean_aggr.propagate(hypergraph.hyperedge_index.flip([0]), x=y, size=(hypergraph.num_edges, hypergraph.num_nodes))
    y = torch.abs((2*y)-1)
    y += alpha * torch.rand_like(y) # add random noise

    # identify nodes to be kept or removed from the dataset
    y[hypergraph.y==1] = -100
    keep_indexes = y[hypergraph.train_mask]
    keep_indexes = keep_indexes.argsort(0, True)[:positive_num]
    keep = torch.ones((y.shape[0],), dtype=torch.bool)
    keep[hypergraph.train_mask & (hypergraph.y!=1)] = False
    train_indices = torch.where(hypergraph.train_mask)[0]
    keep[train_indices[keep_indexes]] = True
    
    # modify the "hypergraph" object by removing the hypernodes
    hypergraph.train_mask = hypergraph.train_mask[keep]
    hypergraph.val_mask = hypergraph.val_mask[keep]
    hypergraph.test_mask = hypergraph.test_mask[keep]
    hypergraph.y = hypergraph.y[keep]
    hypergraph.hyperedge_index = hypergraph.hyperedge_index[:, keep[hypergraph.hyperedge_index[0]]]
    new_indexes = -torch.ones((hypergraph.num_nodes,), dtype=torch.int64)
    hypergraph.num_nodes = int(keep.sum().item())
    new_indexes[keep] = torch.arange(hypergraph.num_nodes)
    hypergraph.hyperedge_index[0] = new_indexes[hypergraph.hyperedge_index[0]]
    return hypergraph

def hyperedge_selection(hypergraph: Hypergraph, l: int = 2, beta: float = 0.5):
    """
    Performs undersampling of the hyperedges in the hypergraph, according to:
    Hypergraph-based importance assessment for binary classification data
    https://link.springer.com/article/10.1007/s10115-022-01786-2
    Except that instead of random walks that approximate node rating, the
    standard mean aggregation and propagation methods are used.
    
    :param hypergraph: Hypergraph to be undersampled.
    :type hypergraph: Hypergraph
    :param l: Number of iterations (aggregation and propagation steps). Equivalent
    to the length of random walks in the original implementation.
    :type l: int
    :param beta: The portion of hyperedges to be kept.
    :type beta: float
    """
    # estimate hyperedge importance scores
    m = HyperedgeAggregation()
    x = hypergraph.y.float().reshape(-1, 1)
    x[~hypergraph.train_mask] = 0
    for _ in range(l):
        x = m.propagate(hypergraph.hyperedge_index, x=x, size=(hypergraph.num_nodes, hypergraph.num_edges))
        x = m.propagate(hypergraph.hyperedge_index.flip([0]), x=x, size=(hypergraph.num_edges, hypergraph.num_nodes))
    x = m.propagate(hypergraph.hyperedge_index, x=x, size=(hypergraph.num_nodes, hypergraph.num_edges))
    x = torch.abs((2*x)-1)
    
    # identify what nodes should be kept
    keep = (x < torch.quantile(x, beta).item()).reshape(-1)
    # modify the "hypergraph" object, keeping only the selected hyperedges
    hypergraph.hyperedge_index = hypergraph.hyperedge_index[:, keep[hypergraph.hyperedge_index[1]]]
    new_indexes = -torch.ones((hypergraph.num_edges,), dtype=torch.int64)
    hypergraph.num_edges = int(keep.sum().item())
    new_indexes[keep] = torch.arange(hypergraph.num_edges)
    hypergraph.hyperedge_index[1] = new_indexes[hypergraph.hyperedge_index[1]]
    if hypergraph.hyperedge_weight!=None:
        hypergraph.hyperedge_weight = hypergraph.hyperedge_weight[keep]
    return hypergraph