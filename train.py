import torch
import torch_geometric
from torchmetrics import AUROC, AveragePrecision
from models import RandomGaussian
from torch_geometric.data import Data
from random import shuffle
from math import ceil
import random

def fit_transform_node2vec(node2vec_model, patience: int = 0, delta: float = 0.0, batch_size: int = 512, num_workers: int = 0):
    last_loss = float("inf")
    current_patience = patience
    optimizer = torch.optim.SparseAdam(list(node2vec_model.parameters()), lr=0.01)
    device = node2vec_model.embedding.weight.device

    loader = node2vec_model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    node2vec_model.train()
    while current_patience>=0:
        current_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            current_loss+=loss.item()
        current_loss /= len(loader)
        if last_loss<current_loss+delta:
            current_patience -= 1
        else:
            current_patience = patience
        last_loss = min(current_loss, last_loss)
    node2vec_model.eval()
    with torch.no_grad():
        x = node2vec_model().detach()
    return x

def train_GNN(model, graph, patience: int = 0, delta: float = 0.0, weight=None):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    # verify the data format, fix existing data, and add missing data
    if not isinstance(graph, torch_geometric.data.Data) or graph.x==None or graph.edge_index==None or graph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y. The properties of the "
    if graph.edge_weight==None:
        graph.edge_weight = torch.ones((graph.edge_index.shape[1],)).to(graph.edge_index.device)
    if graph.train_mask==None:
        graph.train_mask = torch.ones((graph.num_nodes,), dtype=torch.bool).to(graph.x.device)
    if graph.val_mask==None:
        graph.train_mask = torch.ones((graph.num_nodes,), dtype=torch.bool).to(graph.x.device)
    if len(graph.y.shape)>=2 and graph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only."
    if len(graph.y)==1:
        graph.y = graph.y.reshape(-1, 1)
    graph.y = graph.y.float().to(graph.x.device)

    while current_patience>=0:
        model.train()
        optimizer.zero_grad()
        preds = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight)
        loss = loss_fn(preds[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[graph.val_mask], graph.y[graph.val_mask])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    preds = preds.detach()
    return preds[:, 0]

def train_GNN_batches(model, graph, num_neighbors:int = 3, patience: int = 0, delta: float = 0.0, batch_size: int = 512, weight=None):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.005)

    # verify the data format, fix existing data, and add missing data
    if not isinstance(graph, torch_geometric.data.Data) or graph.x==None or graph.edge_index==None or graph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y. The properties of the "
    if graph.edge_weight==None:
        graph.edge_weight = torch.ones((graph.edge_index.shape[1],)).to(graph.edge_index.device)
    if graph.train_mask==None:
        graph.train_mask = torch.ones((graph.num_nodes,), dtype=torch.bool).to(graph.x.device)
    if graph.val_mask==None:
        graph.train_mask = torch.ones((graph.num_nodes,), dtype=torch.bool).to(graph.x.device)
    if len(graph.y.shape)>=2 and graph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only. The second dimension should be 1."
    if len(graph.y)==1:
        graph.y = graph.y.reshape(-1, 1)
    graph.y = graph.y.float().to(graph.x.device)

    loader = torch_geometric.loader.LinkNeighborLoader(graph, batch_size=batch_size, num_neighbors=[num_neighbors], shuffle=True)
    while current_patience>=0:
        model.train()
        for batch_graph in loader:
            batch_graph = batch_graph.to(graph.x.device)
            optimizer.zero_grad()
            preds = model(batch_graph.x, batch_graph.edge_label_index, edge_weight=graph.edge_weight[batch_graph.e_id])
            loss = loss_fn(preds[batch_graph.train_mask], batch_graph.y[batch_graph.train_mask])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight)
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[graph.val_mask], graph.y[graph.val_mask])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
        print(loss.item())
    preds = preds.detach()
    return preds[:, 0]


def train_HGNN(model, hypergraph, x, hyperedge_attr: torch.Tensor = None, patience: int = 0, delta: float = 0.0, weight=None):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    # verify the data format, fix existing data, and add missing data
    if not isinstance(hypergraph, torch_geometric.data.Data) or x==None or hypergraph.hyperedge_index==None or hypergraph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y. The properties of the "
    if hypergraph.hyperedge_weight==None:
        hypergraph.hyperedge_weight = torch.ones((hypergraph.num_edges,)).to(hypergraph.hyperedge_index.device)
    if hypergraph.train_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.hyperedge_index.device)
    if hypergraph.val_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.hyperedge_index.device)
    if len(hypergraph.y.shape)>=2 and hypergraph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only."
    if len(hypergraph.y.shape)==1:
        hypergraph.y = hypergraph.y.reshape(-1, 1)
    if hyperedge_attr==None and model.supports_hyperedge_attr:
        hyperedge_attr = RandomGaussian(x.shape[-1])(hypergraph.num_edges)
    x = x.to(hypergraph.hyperedge_index.device)
    y = hypergraph.y.float().to(hypergraph.hyperedge_index.device)

    while current_patience>=0:
        model.train()
        optimizer.zero_grad()
        kwargs = {}
        if model.supports_hyperedge_attr:
            kwargs["hyperedge_attr"] = hyperedge_attr
        if model.supports_hyperedge_weight:
            kwargs["hyperedge_weight"] = hypergraph.hyperedge_weight
        preds = model(x, hypergraph.hyperedge_index, **kwargs)
        loss = loss_fn(preds[hypergraph.train_mask], y[hypergraph.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[hypergraph.val_mask], y[hypergraph.val_mask])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    preds = preds.detach()
    return preds[:, 0]

def train_HGNN_batches(model, hypergraph, x, num_neighbors:list = 3, hyperedge_attr: torch.Tensor = None, patience: int = 0, delta: float = 0.0, weight=None):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.005)

    # verify the data format, fix existing data, and add missing data
    if not isinstance(hypergraph, torch_geometric.data.Data) or x==None or hypergraph.hyperedge_index==None or hypergraph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y."
    if hypergraph.hyperedge_weight==None:
        hypergraph.hyperedge_weight = torch.ones((hypergraph.num_edges,)).to(hypergraph.hyperedge_index.device)
    if hypergraph.train_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.hyperedge_index.device)
    if hypergraph.val_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.hyperedge_index.device)
    if len(hypergraph.y.shape)>=2 and hypergraph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only."
    if len(hypergraph.y.shape)==1:
        hypergraph.y = hypergraph.y.reshape(-1, 1)
    if hyperedge_attr==None and model.supports_hyperedge_attr:
        hyperedge_attr = RandomGaussian(x.shape[-1])(hypergraph.num_edges)
    x = x.to(hypergraph.hyperedge_index.device)
    y = hypergraph.y.float().to(hypergraph.hyperedge_index.device)

    while current_patience>=0:
        model.train()
        batch_nodes = list(range(hypergraph.num_nodes))
        shuffle(batch_nodes)
        for b in range(ceil(hypergraph.num_nodes/num_neighbors)):
            optimizer.zero_grad()
            b_nodes = batch_nodes[b*num_neighbors:(b+1)*num_neighbors]
            batch_x = x[b_nodes]
            b_nodes_tensor = torch.as_tensor(b_nodes, device=batch_x.device)
            edge_index_filter = torch.isin(
                hypergraph.hyperedge_index[0],
                b_nodes_tensor
            )
            edge_index = hypergraph.hyperedge_index[:, edge_index_filter]
            min_hyperedge_index = edge_index[1].min()
            edge_index[1] -= min_hyperedge_index
            remapping = torch.full((hypergraph.num_nodes,), -1, dtype=torch.long, device=edge_index.device)
            remapping[b_nodes] = torch.arange(len(b_nodes), device=edge_index.device)
            edge_index[0] = remapping[edge_index[0]]
            kwargs = {}
            if model.supports_hyperedge_attr:
                kwargs["hyperedge_attr"] = hyperedge_attr[min_hyperedge_index:]
            if model.supports_hyperedge_weight:
                kwargs["hyperedge_weight"] = hypergraph.hyperedge_weight[min_hyperedge_index:]
            preds = model(x[b_nodes], edge_index, **kwargs)
            batch_train_mask = hypergraph.train_mask[b_nodes]
            loss = loss_fn(preds[batch_train_mask], y[b_nodes][batch_train_mask])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = model(x, hypergraph.hyperedge_index)
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[hypergraph.val_mask], y[hypergraph.val_mask])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    preds = preds.detach()
    return preds[:, 0]

def evaluate(preds, target):
    roc_auc = AUROC('binary')(preds, target)
    pr_auc = AveragePrecision('binary')(preds, target)
    return roc_auc, pr_auc