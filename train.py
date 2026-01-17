import torch
import torch_geometric
from torchmetrics import AUROC, PrecisionRecallCurve
from sklearn.metrics import auc, roc_curve

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

def train_GNN(model, graph, patience: int = 0, delta: float = 0.0):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss()
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
    preds = preds.detach()[graph.val_mask]
    return preds[:, 0]

def train_GNN_batches(model, graph, num_neighbors:list = [3], patience: int = 0, delta: float = 0.0, batch_size: int = 512, num_workers: int = 0):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss()
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
        assert "The current function was designed for binary classes only."
    if len(graph.y)==1:
        graph.y = graph.y.reshape(-1, 1)
    graph.y = graph.y.float().to(graph.x.device)

    loader = torch_geometric.loader.LinkNeighborLoader(graph, batch_size=batch_size, num_neighbors=num_neighbors,
                                                       num_workers=num_workers, shuffle=True)
    while current_patience>=0:
        model.train()
        for batch_graph in loader:
            optimizer.zero_grad()
            preds = model(batch_graph.x, batch_graph.edge_index, edge_weight=batch_graph.edge_weight)
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
    preds = preds.detach()[graph.val_mask]
    return preds[:, 0]


def train_HGNN(model, hypergraph, x, patience: int = 0, delta: float = 0.0):
    last_loss = float("inf")
    current_patience = patience
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    # verify the data format, fix existing data, and add missing data
    if not isinstance(hypergraph, torch_geometric.data.Data) or x==None or hypergraph.edge_index==None or hypergraph.y==None:
        assert "Graph component must be a torch_geometric.data.Data object with attributes: x, edge_index, and y. The properties of the "
    if hypergraph.edge_weight==None:
        hypergraph.edge_weight = torch.ones((hypergraph.edge_index.shape[1],)).to(hypergraph.edge_index.device)
    if hypergraph.train_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.x.device)
    if hypergraph.val_mask==None:
        hypergraph.train_mask = torch.ones((hypergraph.num_nodes,), dtype=torch.bool).to(hypergraph.x.device)
    if len(hypergraph.y.shape)>=2 and hypergraph.y.shape[1]!=1:
        assert "The current function was designed for binary classes only."
    if len(hypergraph.y)==1:
        hypergraph.y = hypergraph.y.reshape(-1, 1)
    y = hypergraph.y.float().to(hypergraph.x.device)

    while current_patience>=0:
        model.train()
        optimizer.zero_grad()
        preds = model(x, hypergraph.hyperedge_index, edge_weight=hypergraph.hyperedge_weight)
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
    preds = preds.detach()[hypergraph.val_mask]
    return preds[:, 0]

def evaluate(preds, target):
    roc_auc = AUROC('binary')(preds, target)
    pr_curve = PrecisionRecallCurve(task="binary")
    precision, recall, _ = pr_curve(preds, target)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc