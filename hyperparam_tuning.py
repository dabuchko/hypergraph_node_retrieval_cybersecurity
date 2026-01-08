import argparse
import random
import threading
import json
import os
import datetime
import re
import copy
from time import time

import torch
import torch_geometric
from torchmetrics import AUROC, PrecisionRecallCurve

import numpy as np
from sklearn.metrics import auc, roc_curve

from data import *
from models import *

class HyperparameterSetGenerator():
    """
    Iterator that generates sets of hyperparameters based on the provided ranges.
    If it is possible to create at most 1 million unique combinations of hyperparameters
    in the provided ranges, then hyperparameters are generated for Grid Search and
    the iterator will eventually halt, after returning all possible combinations.
    If more than 1 million of combinations of hyperparameters are possible, than
    the hyperparameters are returned for Random Search, hyperparameter combinations
    are returned infinitely and may repeat with an extremelly small chance (assuming
    more than 1 million combinations are possible).
    For this reason during iterating over hyperparameter combinations, some
    additional stopping criteria should be defined. For example, time limit or
    more realistic maximum of number of hyperparameter combinations,
    because one million is unrealistically high even for trivial algorithms.
    """
    def __init__(self, *hyperparameters_ranges):
        self._lock = threading.Lock()
        self._hyperparam_ranges = hyperparameters_ranges
        # if number of possible combinations is less than one
        # million, then precompute and shuffle all possible
        # hyperparameter combinations, otherwise just sample
        # combinations randomly upon request. This is done
        # to account for the case when number of possible
        # combinations is small enough to reach the end
        # during the enumeration
        self._precomputed = None
        number_of_possible_combinations = 1
        for hr in hyperparameters_ranges:
            for values in hr.values():
                number_of_possible_combinations *= len(values)
                if number_of_possible_combinations > 1_000_000:
                    break
        if number_of_possible_combinations < 1_000_000:
            self._precomputed = [[dict() for _ in range(len(hyperparameters_ranges))]]
            for hr_n, hr in enumerate(hyperparameters_ranges):
                for key in hr.keys():
                    precomputed = self._precomputed
                    self._precomputed = []
                    for value in hr[key]:
                        for comb in precomputed:
                            comb_copy = copy.deepcopy(comb)
                            comb_copy[hr_n][key] = value
                            self._precomputed.append(comb_copy)
            random.shuffle(self._precomputed)
            self._counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._precomputed==None:
                # return randomly sampled combination
                comb = [dict() for _ in range(len(self._hyperparam_ranges))]
                for hr_n, hr in enumerate(self._hyperparam_ranges):
                    for key in hr.keys():
                        comb[hr_n][key] = random.choice(hr[key])
                return comb
            else:
                self._counter += 1
                if self._counter>=len(self._precomputed):
                    raise StopIteration
                return self._precomputed[self._counter - 1]


def main(args: argparse.ArgumentParser):
    # verify correct combination of methods
    embedding_set = args.embedding!=None
    features_set = args.feature_based!=None
    graph_set = args.graph_based!=None
    if not ((features_set and not graph_set) or
        (not features_set and graph_set)):
        raise Exception("Only one option is available. Either provide 'feature_based' or 'graph_based' argument.")

    # Set the random seed and the number of threads.
    if args.seed is not None:
        torch_geometric.seed.seed_everything(args.seed) # includes PyTorch, NumPy, and general Python seeding

    if args.threads is not None and args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
    
    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # create dataset
    data = DATASETS[args.dataset]()

    # load hyperparameters ranges
    try:
        with open(args.hyperparam_ranges, 'r') as file:
            hyperparameters_ranges = json.load(file)
    except:
        raise Exception(f"Failed to load JSON data from {args.hyperparam_ranges}.")
    # initialize HyperparameterSetGenerator
    hp_ranges_loaded = []
    if embedding_set:
        hp_ranges_loaded.append(hyperparameters_ranges[args.embedding])
    else:
        hp_ranges_loaded.append({})
    if features_set:
        hp_ranges_loaded.append(hyperparameters_ranges[args.feature_based])
    if graph_set:
        hp_ranges_loaded.append(hyperparameters_ranges[args.graph_based])
    hp_generator = HyperparameterSetGenerator(*hp_ranges_loaded)


    best_pr_auc = 0
    best_hyperparameters = None
    global_start = int(time())
    val_y = data.labels[data.val_mask]

    for embedding_hp_set, method_hp_set in hp_generator:
        print("started")
        local_start = int(time())
        # compute embeddings if any
        if embedding_set:
            if args.embedding=="Node2Vec":
                embedding_hp_set["edge_index"] = data.incidence_graph().edge_index.long()
                embedding_hp_set["num_nodes"] = int(embedding_hp_set["edge_index"].max()) + 1
                breakpoint()
                node2vec = EMBEDDING_METHODS[args.embedding](**embedding_hp_set).to(device)
                del embedding_hp_set["edge_index"]
                del embedding_hp_set["num_nodes"]
                loader = node2vec.loader(batch_size=args.batch_size, shuffle=True, num_workers=3)
                optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=args.lr)
                node2vec.train()
                patience = args.patience
                last_loss = 0
                current_loss = float("inf")
                while patience>=0:
                    last_loss = current_loss
                    current_loss = 0
                    for pos_rw, neg_rw in loader:
                        optimizer.zero_grad()
                        loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                        loss.backward()
                        optimizer.step()
                        current_loss+=loss.item()
                    current_loss /= len(loader)
                    if last_loss<current_loss-args.delta:
                        patience+=1
                    else:
                        patience = args.patience
                with torch.no_grad():
                    x = node2vec()[:data.num_nodes]
            else:
                embedding_class = EMBEDDING_METHODS[args.embedding](**embedding_hp_set)
                if isinstance(embedding_class, RandomGaussian):
                    x = embedding_class(data.num_nodes)
                elif isinstance(embedding_class, MatrixFactorization):
                    x, _ = embedding_class(data.sparse_incidence_matrix(), data.hyperedge_weight)
                elif isinstance(embedding_class, SpectralEmbedding):
                    incidence_graph = data.incidence_graph()
                    x = embedding_class(incidence_graph.edge_index, incidence_graph.edge_weight)[:data.num_nodes]
        elif args.graph_based!="Label Propagation" and args.graph_based!="CSP":
            x = data.incidence_matrix()
        else:
            x = None
        
        embedding_end = int(time())
        # train methods on embeddings (if any) and generate predictions 'preds'
        train_x = x[data.train_mask]
        train_y = data.labels[data.train_mask]
        val_x = x[data.val_mask]
        if features_set:
            train_x = train_x.numpy()
            train_y = train_y.numpy()
            val_x = val_x.numpy()
            method = FEATURE_METHODS[args.feature_based](**method_hp_set).fit(train_x, train_y)
            preds = torch.tensor(method.predict_proba(val_x))[:, 1]
        else:
            if args.graph_based in GRAPH_METHODS:
                if args.graph_representation=="incidence":
                    graph = data.incidence_graph()
                    edge_index = graph.edge_index
                    edge_weight = graph.edge_weight
                    y = torch.cat([graph.hypergraph_nodes.y, torch.zeros((graph.num_edges))], 0)
                    train_mask = torch.cat([graph.hypergraph_nodes.train_mask, torch.zeros((graph.num_edges))], 0)
                    val_mask = torch.cat([graph.hypergraph_nodes.val_mask, torch.zeros((graph.num_edges))], 0)
                    test_mask = torch.cat([graph.hypergraph_nodes.test_mask, torch.zeros((graph.num_edges))], 0)
                else:
                    graph = data.clique_graph()
                    edge_index = graph.edge_index
                    edge_weight = graph.edge_weight
                    y = graph.y
                    train_mask = graph.train_mask
                    val_mask = graph.val_mask,
                    test_mask = graph.test_mask
                method_hp_set["out_channels"] = 1
                model = GRAPH_METHODS[args.graph_based](**method_hp_set)
                
            elif args.graph_based in HYPERGRAPH_METHODS:
                pass

            last_loss = float("inf")
            patience = args.patience
            loss_fn = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), 0.01)
            while patience>=0:
                model.train()
                optimizer.zero_grad()
                preds = model(x, edge_index, edge_weight=edge_weight)
                loss = loss_fn(preds[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
                loss = loss_fn(preds[val_mask], y[train_mask])
                if last_loss-loss<args.delta:
                    patience-=1
                else:
                    patience = args.patience
            preds = preds.detach()
        local_end = int(time())

        if int(time())-global_start>args.time_limit:
            break
        # evaluate predictions using Accuracy, ROC-AUC and PR-AUC metrics
        threshold = data.labels[data.train_mask].float().mean()
        accuracy = (val_y==(preds>threshold)).float().mean()
        roc_auc = AUROC('binary')(preds, val_y)
        pr_curve = PrecisionRecallCurve(task="binary")
        precision, recall, _ = pr_curve(preds, val_y)
        pr_auc = auc(recall, precision)
        # save results
        os.makedirs(args.logdir, exist_ok=True) # it is important to create directory not earlier than at least one iteration succeeded
        with open(args.logdir + "/results.csv", "a+") as f:
            embedding_time = embedding_end - local_start
            total_time = local_end - local_start
            f.write(f"{accuracy},{roc_auc},{pr_auc},{embedding_time},{total_time},{embedding_hp_set},{method_hp_set}\n")
        if pr_auc>best_pr_auc:
            best_pr_auc = pr_auc
            best_hyperparameters = (embedding_hp_set, method_hp_set)
            # save best predictions
            if os.path.isfile(args.logdir+"/preds.pt"):
                os.remove(args.logdir+"/preds.pt")
            torch.save(preds, args.logdir+"/preds.pt")
    print(best_hyperparameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys(), type=str,
                        help="Datasets on which hyperparameter tuning should be run.")
    parser.add_argument("--embedding", choices=EMBEDDING_METHODS.keys(), type=str,
                        help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--feature_based", choices=FEATURE_METHODS.keys(), type=str,
                        help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--graph_based", choices=list(GRAPH_METHODS.keys())+list(HYPERGRAPH_METHODS.keys()),
                        type=str, help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--graph_representation", default="incidence", choices=["incidence", "clique"],
                        type=str, help="Graph representation of the hypergraph that should be used." \
                        "Ignored unless 'graph_based' contains method that operate on graphs.")
    parser.add_argument("--delta", default=0.1, type=float, help="Only loss change greater than delta"
    "may be counted as positive improvement, otherwise early stopping may be applied.")
    parser.add_argument("--patience", default=20, type=int, help="Maximum number of epochs for which" \
    "loss decrease less than delta may be tollerated.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for Adam optimizer.")
    parser.add_argument("--hyperparam_ranges", default="hyperparam_ranges.json",
                        type=str, help="Path to the JSON file containing hyperparameter ranges" \
                        "for each method.")
    parser.add_argument("--logdir", type=str, default="logs", help="Default directory for storing logs.")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum time in seconds to spend on hyperparameter tuning.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
