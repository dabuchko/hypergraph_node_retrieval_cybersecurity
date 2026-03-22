import torch
from data import Hypergraph
from models.hypergraph.unignn import HyperedgeAggregation

def hyper_undersampling(hypergraph: Hypergraph, l: int = 3, alpha: float = 0.1):
    m = HyperedgeAggregation()
    x = hypergraph.y.float().reshape(-1, 1)
    x[~hypergraph.train_mask] = 0
    true_num = int(x.sum().item())
    for _ in range(l):
        x = m.propagate(hypergraph.hyperedge_index, x=x, size=(hypergraph.num_nodes, hypergraph.num_edges))
        x = m.propagate(hypergraph.hyperedge_index.flip([0]), x=x, size=(hypergraph.num_edges, hypergraph.num_nodes))
    x = torch.abs((2*x)-1)
    x += alpha * torch.rand_like(x)
    x[hypergraph.y==1] = -100
    keep_indexes = x[hypergraph.train_mask]
    keep_indexes = keep_indexes.argsort(0, True)[:true_num]
    keep = torch.ones((x.shape[0],), dtype=torch.bool)
    keep[hypergraph.train_mask & (hypergraph.y!=1)] = False
    train_indices = torch.where(hypergraph.train_mask)[0]
    keep[train_indices[keep_indexes]] = True
    
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
    m = HyperedgeAggregation()
    x = hypergraph.y.float().reshape(-1, 1)
    x[~hypergraph.train_mask] = 0
    for _ in range(l):
        x = m.propagate(hypergraph.hyperedge_index, x=x, size=(hypergraph.num_nodes, hypergraph.num_edges))
        x = m.propagate(hypergraph.hyperedge_index.flip([0]), x=x, size=(hypergraph.num_edges, hypergraph.num_nodes))
    x = m.propagate(hypergraph.hyperedge_index, x=x, size=(hypergraph.num_nodes, hypergraph.num_edges))
    x = torch.abs((2*x)-1)
    
    keep = (x < torch.quantile(x, beta).item()).reshape(-1)
    
    hypergraph.hyperedge_index = hypergraph.hyperedge_index[:, keep[hypergraph.hyperedge_index[1]]]
    new_indexes = -torch.ones((hypergraph.num_edges,), dtype=torch.int64)
    hypergraph.num_edges = int(keep.sum().item())
    new_indexes[keep] = torch.arange(hypergraph.num_edges)
    hypergraph.hyperedge_index[1] = new_indexes[hypergraph.hyperedge_index[1]]
    return hypergraph