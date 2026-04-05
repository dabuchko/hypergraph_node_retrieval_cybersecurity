import torch
import torch_geometric
from torchmetrics import AUROC, AveragePrecision
from models import RandomGaussian
from random import shuffle
from math import ceil
from data.hypergraph import Hypergraph

def fit_transform_node2vec(node2vec_model, batch_size: int = 512, num_workers: int = 0, device="cpu"):
    device = torch.device(device)
    node2vec_model.to(device)
    optimizer = torch.optim.Adam(list(node2vec_model.parameters()), lr=0.01)

    loader = node2vec_model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    node2vec_model.train()
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
    node2vec_model.eval()
    with torch.no_grad():
        x = node2vec_model().detach()
    return x

def verify_and_preprocess_graph(graph):
    if not isinstance(graph, torch_geometric.data.Data) or graph.x==None or graph.edge_index==None or graph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y."
    if len(graph.y.shape)>=2 and graph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only."
    if graph.edge_weight==None:
        graph.edge_weight = torch.ones((graph.edge_index.shape[1],))
    if graph.train_mask==None:
        graph.train_mask = torch.ones((graph.num_nodes,), dtype=torch.bool)
    if graph.val_mask==None:
        graph.val_mask = torch.ones((graph.num_nodes,), dtype=torch.bool)
    graph.y = graph.y.float()
    if len(graph.y.shape)==1:
        graph.y = graph.y.reshape(-1, 1)
    return graph

def train_GNN(model, graph, patience: int = 0, delta: float = 0.0, weight=None, device="cpu"):
    device = torch.device(device)
    model.to(device)
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    # verify the data format, fix existing data, and add missing data
    graph = verify_and_preprocess_graph(graph)

    while current_patience>=0:
        model.train()
        optimizer.zero_grad()
        preds = model(graph.x.to(device), graph.edge_index.to(device), edge_weight=graph.edge_weight.to(device))
        loss = loss_fn(preds[graph.train_mask.to(device)], graph.y.to(device)[graph.train_mask.to(device)])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[graph.val_mask.to(device)], graph.y.to(device)[graph.val_mask.to(device)])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    preds = preds.detach()
    return preds[:, 0]

def train_GNN_batches(model, graph, num_neighbors:int = 3, patience: int = 0, delta: float = 0.0, batch_size: int = 512, num_workers: int = 0, weight=None, device="cpu"):
    device = torch.device(device)
    model.to(device)
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.005)

    # verify the data format, fix existing data, and add missing data
    graph = verify_and_preprocess_graph(graph)
    loader = torch_geometric.loader.LinkNeighborLoader(graph, batch_size=batch_size, num_neighbors=[num_neighbors], shuffle=True, num_workers=num_workers)

    while current_patience>=0:
        model.train()
        for batch_graph in loader:
            if batch_graph.train_mask.sum()==0:
                continue
            batch_graph = batch_graph.to(device)
            optimizer.zero_grad()
            batch_edge_weight = graph.edge_weight[batch_graph.e_id.to(graph.edge_weight.device)].to(device)
            batch_preds = model(batch_graph.x, batch_graph.edge_label_index, edge_weight=batch_edge_weight)
            loss = loss_fn(batch_preds[batch_graph.train_mask], batch_graph.y[batch_graph.train_mask])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = torch.tensor(0.0, device=device)
            preds = torch.zeros_like(graph.y, device=device)
            model.eval()
            for batch_graph in loader:
                batch_graph = batch_graph.to(device)
                batch_edge_weight = graph.edge_weight[batch_graph.e_id.to(graph.edge_weight.device)].to(device)
                batch_preds = model(batch_graph.x, batch_graph.edge_label_index, edge_weight=batch_edge_weight)
                if batch_graph.val_mask.sum()!=0:
                    loss += loss_fn(batch_preds[batch_graph.val_mask], batch_graph.y[batch_graph.val_mask])
                preds[batch_graph.n_id] = batch_preds
            loss /= len(loader)
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    return preds[:, 0]

def verify_and_preprocess_hypergraph(hypergraph, x):
    if (not isinstance(hypergraph, Hypergraph)) or x==None or hypergraph.hyperedge_index==None or hypergraph.y==None:
        assert "Hypergraph component must be a data.hypergraph.Hypergraph object with attributes: x, hyperedge_index, and y."
    if hypergraph.hyperedge_weight==None:
        hypergraph.hyperedge_weight = torch.ones((hypergraph.num_edges,))
    if hypergraph.train_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool)
    if hypergraph.val_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool)
    if len(hypergraph.y.shape)>=2 and hypergraph.y.shape[1]!=1:
        assert "This function was designed for binary classes only."
    if len(hypergraph.y.shape)==1:
        hypergraph.y = hypergraph.y.reshape(-1, 1)
    hypergraph.y = hypergraph.y.float()
    return hypergraph

def train_HGNN(model, hypergraph, x, hyperedge_attr: torch.Tensor = None, patience: int = 0, delta: float = 0.0, weight=None, device="cpu"):
    device = torch.device(device)
    model.to(device)
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    # verify the data format, fix existing data, and add missing data
    hypergraph = verify_and_preprocess_hypergraph(hypergraph, x)
    if hyperedge_attr==None and model.supports_hyperedge_attr:
        hyperedge_attr = RandomGaussian(x.shape[-1])(hypergraph.num_edges)

    while current_patience>=0:
        model.train()
        optimizer.zero_grad()
        kwargs = {}
        if model.supports_hyperedge_attr:
            kwargs["hyperedge_attr"] = hyperedge_attr.to(device)
        if model.supports_hyperedge_weight:
            kwargs["hyperedge_weight"] = hypergraph.hyperedge_weight.to(device)
        preds = model(x.to(device), hypergraph.hyperedge_index.to(device), **kwargs)
        loss = loss_fn(preds[hypergraph.train_mask.to(device)], hypergraph.y.to(device)[hypergraph.train_mask.to(device)])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(preds[hypergraph.val_mask.to(device)], hypergraph.y.to(device)[hypergraph.val_mask.to(device)])
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
    preds = preds.detach()
    return preds[:, 0]

class HGNN_batch_sampler(torch.utils.data.Sampler):
    def __init__(self, hypergraph, batch_size = 64, val = False):
        self.hypergraph = hypergraph
        self.val = val
        self.batch_size = batch_size
        self.shuffle()
    def shuffle(self):
        self.order = torch.argsort(torch.rand((self.hypergraph.num_nodes,))[self.hypergraph.hyperedge_index[0]], stable=True).tolist()
    def collate(self, batch_hyperedge_index):
        batch_hyperedge_index = torch.stack(batch_hyperedge_index).T
        batch_hypernodes, batch_hyperedge_index[0] = torch.unique(batch_hyperedge_index[0], return_inverse=True)
        batch_hyperedges, batch_hyperedge_index[1] = torch.unique(batch_hyperedge_index[1], return_inverse=True)
        if self.val:
            batch_mask = self.hypergraph.val_mask[batch_hypernodes]
        else:
            batch_mask = self.hypergraph.train_mask[batch_hypernodes]
        return batch_hypernodes, batch_hyperedges, batch_hyperedge_index, batch_mask
    def __len__(self):
        return ceil(self.hypergraph.hyperedge_index.shape[1]/self.batch_size)
    def __iter__(self):
        for i in range(0, self.hypergraph.hyperedge_index.shape[1], self.batch_size):
            yield self.order[i:i+self.batch_size]
    def train(self):
        self.val = False
    def eval(self):
        self.val = True

def train_HGNN_batches(model, hypergraph, x, batch_size:list = 64, num_workers: int = 0, hyperedge_attr: torch.Tensor = None, patience: int = 0, delta: float = 0.0, weight=None, device="cpu"):
    device = torch.device(device)
    model.to(device)
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss(weight)
    optimizer = torch.optim.Adam(model.parameters(), 0.005)

    # verify the data format, fix existing data, and add missing data
    hypergraph = verify_and_preprocess_hypergraph(hypergraph, x)
    if hyperedge_attr==None and model.supports_hyperedge_attr:
        hyperedge_attr = RandomGaussian(x.shape[-1])(hypergraph.num_edges)
    
    batch_sampler = HGNN_batch_sampler(hypergraph, batch_size)
    dataloader = torch.utils.data.DataLoader(hypergraph.hyperedge_index.T, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=batch_sampler.collate)
    while current_patience>=0:
        batch_sampler.train()
        model.train()
        for batch_hypernodes, batch_hyperedges, batch_hyperedge_index, batch_train_mask in dataloader:
            if batch_train_mask.sum()==0:
                continue
            optimizer.zero_grad()
            kwargs = {}
            if model.supports_hyperedge_attr:
                kwargs["hyperedge_attr"] = hyperedge_attr[batch_hyperedges].to(device)
            if model.supports_hyperedge_weight:
                kwargs["hyperedge_weight"] = hypergraph.hyperedge_weight[batch_hyperedges].to(device)
            batch_preds = model(x[batch_hypernodes].to(device), batch_hyperedge_index.to(device), **kwargs)
            loss = loss_fn(batch_preds[batch_train_mask.to(device)], hypergraph.y[batch_hypernodes][batch_train_mask].to(device))
            loss.backward()
            optimizer.step()
        
        batch_sampler.eval()
        model.eval()
        with torch.no_grad():
            loss = torch.tensor(0.0, device=device)
            preds = torch.zeros_like(hypergraph.y, device=device)
            for batch_hypernodes, batch_hyperedges, batch_hyperedge_index, batch_val_mask in dataloader:
                kwargs = {}
                if model.supports_hyperedge_attr:
                    kwargs["hyperedge_attr"] = hyperedge_attr[batch_hyperedges].to(device)
                if model.supports_hyperedge_weight:
                    kwargs["hyperedge_weight"] = hypergraph.hyperedge_weight[batch_hyperedges].to(device)
                batch_preds = model(x[batch_hypernodes].to(device), batch_hyperedge_index.to(device), **kwargs)
                if batch_val_mask.sum()!=0:
                    loss += loss_fn(batch_preds[batch_val_mask.to(device)], hypergraph.y[batch_hypernodes][batch_val_mask].to(device))
                preds[batch_hypernodes] = batch_preds
            loss /= ceil(hypergraph.hyperedge_index.shape[1] / batch_size)
        if last_loss-loss<delta:
            current_patience-=1
        else:
            current_patience = patience
        last_loss = min(loss.item(), last_loss)
        batch_sampler.shuffle()
    preds = preds.detach()
    return preds[:, 0]

def evaluate(preds, target):
    roc_auc = AUROC('binary')(preds, target)
    pr_auc = AveragePrecision('binary')(preds, target)
    return roc_auc, pr_auc