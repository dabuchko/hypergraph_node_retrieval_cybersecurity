import argparse
import random
import threading
import json
import os
import datetime
import re
import copy
import gc
import traceback
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE
from time import time

import torch
import torch_geometric
import numpy as np

from data import *
from models import *
from hypergraph_undersampling import *

from train import train_GNN, train_GNN_batches, train_HGNN, train_HGNN_batches, evaluate, train_predict_node2vec

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
    def __init__(self, *hyperparameters_ranges: dict):
        """
        Initializes the HyperparameterSetGenerator with the provided hyperparameter ranges.
        
        :param hyperparameters_ranges: Set of dictionaries, where each dictionary corresponds
        to the set of hyperparameters for a specific method. The keys of the dictionary are
        hyperparameter names, values are lists of possible hyperparameter values.
        :type hyperparameters_ranges: dict
        """
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
        """
        Returns the iterator over hyperparameter combinations. Terminates if end is
        reached and the number of all possible combinations is smaller than one million,
        otherwise samples random combinations indefinitely (may sample duplicates in this case).
        """
        return self

    def __next__(self):
        """
        Returns the next combination of hyperparameters. Raise StopIteration exception if end is
        reached and the number of all possible combinations is smaller than one million,
        otherwise samples random combinations indefinitely (may sample duplicates in this case).
        """
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
                if self._counter>len(self._precomputed):
                    raise StopIteration
                return self._precomputed[self._counter - 1]


def main(args: argparse.ArgumentParser):
    # verify correct combination of methods and learning strategies
    embedding_set = args.embedding!=None
    features_set = args.feature_based!=None
    graph_set = args.graph_based!=None
    if not ((features_set and not graph_set) or
        (not features_set and graph_set)):
        raise Exception("Only one option is available. Either provide 'feature_based' or 'graph_based' argument.")
    if args.train_strategy in ["random_oversampling", "random_undersampling", "SMOTE", "tomek_links"] and  graph_set:
        raise Exception("Specified training strategy is unavailable for graph methods.")
    if args.train_strategy=="weight" and (args.feature_based=="KNN" or args.graph_based=="Label Propagation" or args.graph_based=="CSP"):
        raise Exception("'weight' training strategy is unavailable for KNN and label propagation methods.")
    if args.embedding=="Trainable Embeddings" and features_set:
        raise Exception("'Trainable Embeddings' cannot be used with feature-based methods.")
    if args.embedding=="Trainable Embeddings" and (args.graph_based=="Label Propagation" or args.graph_based=="CSP"):
        raise Exception("'Trainable Embeddings' cannot be used with label propagation methods.")

    # Set the random seed and the number of threads.
    if args.seed is not None:
        torch_geometric.seed.seed_everything(args.seed) # includes PyTorch, NumPy, and general Python seeding

    if args.threads is not None and args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except:
            pass
    
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
    
    # construct dataset
    data = DATASETS[args.dataset]()
    weight_true_class = None
    if args.train_strategy=="weight":
        weight_true_class = data.y.sum().float()
        weight_true_class /= data.y.shape[0] - weight_true_class
        weight_true_class = weight_true_class
    elif args.train_strategy=="hyper_undersampling":
        data = hyper_undersampling(data)
    elif args.train_strategy=="hyperedge_selection":
        data = hyperedge_selection(data)

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
    global_start = time()
    val_y = data.y[data.val_mask]
    test_y = data.y[data.test_mask]

    for embedding_hp_set, method_hp_set in hp_generator:
        # for each hyperparameter combination, generate embeddings (if needed),
        # perform training (if needed), generate predictions and evaluate them.
        # continue until all combinations are exhausted or time limit is reached
        try:
            print(embedding_hp_set, method_hp_set)
            local_start = time()
            embedding_graph = None
            graph = None
            x = None

            # load the graph representation of the hypergraph if needed
            if args.embedding=="Node2Vec" or args.embedding=="Spectral Embedding":
                if args.graph_repr_embedding=="incidence":
                    embedding_graph = data.incidence_graph().to_homogeneous()
                elif args.graph_repr_embedding=="clique":
                    embedding_graph = data.clique_graph()
                embedding_graph.y = embedding_graph.y.float().reshape(-1, 1)
            if args.graph_based in GRAPH_METHODS:
                if embedding_graph!=None and args.graph_repr_GNN==args.graph_repr_embedding:
                    graph = embedding_graph
                else:
                    if args.graph_repr_GNN=="incidence":
                        graph = data.incidence_graph().to_homogeneous()
                    elif args.graph_repr_GNN=="clique":
                        graph = data.clique_graph()
                    graph.y = graph.y.float().reshape(-1, 1)


            # compute embeddings if needed
            if embedding_set:
                if args.embedding=="Node2Vec":
                    embedding_hp_set_copy = embedding_hp_set.copy()
                    del embedding_hp_set_copy["batch_size"]
                    node2vec = EMBEDDING_METHODS[args.embedding](edge_index=embedding_graph.edge_index,
                                                                 num_nodes=embedding_graph.num_nodes,
                                                                 **embedding_hp_set_copy).to(device)
                    x = train_predict_node2vec(node2vec, embedding_hp_set["batch_size"], args.num_workers, device).cpu()
                elif args.embedding=="Trainable Embeddings":
                    x = EMBEDDING_METHODS[args.embedding](data.num_nodes, embedding_hp_set["dim"])
                else:
                    embedding_class = EMBEDDING_METHODS[args.embedding](**embedding_hp_set)
                    if isinstance(embedding_class, RandomGaussian):
                        x = embedding_class(data.num_nodes)
                    elif isinstance(embedding_class, MatrixFactorization):
                        hyperedge_weight = None
                        if data.hyperedge_weight!=None:
                            hyperedge_weight = data.hyperedge_weight.to(device)
                        P, Q = embedding_class(data.sparse_incidence_matrix().to(device), hyperedge_weight)
                        x = torch.cat([P.cpu(), Q.cpu()], 0)
                    elif isinstance(embedding_class, SpectralEmbedding):
                        graph_weight = None
                        if embedding_graph.edge_weight!=None:
                            graph_weight = embedding_graph.edge_weight
                        x = embedding_class(embedding_graph.edge_index, graph_weight, embedding_graph.num_nodes)
                    x = x.to(torch.float)
                if embedding_graph!=None:
                    del embedding_graph
            elif args.graph_based=="CSP":
                x = data.y.clone()
                x[~data.train_mask] = 0 # consider only training labels
            elif args.graph_based=="Label Propagation":
                x = graph.y.clone()
                x[~graph.train_mask] = 0 # consider only training labels
            
            
            embedding_end = time()
            # train methods on embeddings (if any) and generate predictions 'preds'
            if features_set:
                # feature-based methods
                x = x[:data.num_nodes].numpy() # use only hypernode features
                train_x = x[data.train_mask.numpy()]
                train_y = data.y[data.train_mask].numpy()
                method = FEATURE_METHODS[args.feature_based](**method_hp_set)
                if args.train_strategy=="weight":
                    weights = np.ones((train_x.shape[0],), dtype=float)
                    weights[train_y==1] = weight_true_class.numpy()
                    method.fit(train_x, train_y, sample_weight=weights)
                else:
                    # feature based imbalance handling strategies
                    if args.train_strategy=="random_oversampling":
                        train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)
                    elif args.train_strategy=="random_undersampling":
                        train_x, train_y = RandomUnderSampler().fit_resample(train_x, train_y)
                    elif args.train_strategy=="SMOTE":
                        train_x, train_y = SMOTE().fit_resample(train_x, train_y)
                    elif args.train_strategy=="tomek_links":
                        train_x, train_y = TomekLinks().fit_resample(train_x, train_y)
                    method.fit(train_x, train_y)
                preds = torch.tensor(method.predict_proba(x))[:, 1]
            elif args.graph_based=="Label Propagation":
                graph_weight = None
                if graph.edge_weight!=None:
                    graph_weight = graph.edge_weight.to(device)
                preds = GRAPH_METHODS[args.graph_based](**method_hp_set)(x.to(device), graph.edge_index.to(device), edge_weight=graph_weight)
                preds = preds[:data.num_nodes].reshape(-1)
            elif args.graph_based=="CSP":
                preds = HYPERGRAPH_METHODS[args.graph_based](**method_hp_set)(x.to(device), data.hyperedge_index.to(device)).reshape(-1)
            elif args.graph_based in GRAPH_METHODS:
                # graph based methods
                if x is not None:
                    if args.graph_repr_GNN=="incidence":
                        if x.shape[0]!=data.num_nodes + data.num_edges:
                            x.data = torch.cat([x.data, torch.zeros((data.num_nodes + data.num_edges - x.shape[0], x.shape[1]))], 0)
                    else:
                        x.data = x.data[:data.num_nodes]
                    method_hp_set["in_channels"] = x.shape[-1]
                method_hp_set["out_channels"] = 1
                if "batch_size" in method_hp_set.keys():
                    method_hp_set_copy = method_hp_set.copy()
                    del method_hp_set_copy["num_neighbors"]
                    del method_hp_set_copy["batch_size"]
                    model = GRAPH_METHODS[args.graph_based](**method_hp_set_copy)
                    preds = train_GNN_batches(model, graph, x, method_hp_set["num_neighbors"], args.patience,
                                        args.delta, method_hp_set["batch_size"], args.num_workers, weight_true_class, device)
                else:
                    model = GRAPH_METHODS[args.graph_based](**method_hp_set)
                    preds = train_GNN(model, graph, x, args.patience, args.delta, weight_true_class, device)
            elif args.graph_based in HYPERGRAPH_METHODS:
                # hypergraph based methods
                hyperedge_attr = None
                if x is not None:
                    if x.shape[0]>data.num_nodes:
                        hyperedge_attr = x[data.num_nodes:]
                    x.data = x.data[:data.num_nodes]
                    method_hp_set["in_channels"] = x.shape[-1]
                if args.graph_based=="SumMin":
                    method_hp_set["num_nodes"] = data.num_nodes
                method_hp_set["out_channels"] = 1
                if "batch_size" in method_hp_set.keys():
                    method_hp_set_copy = method_hp_set.copy()
                    del method_hp_set_copy["batch_size"]
                    del method_hp_set_copy["sample"]
                    model = HYPERGRAPH_METHODS[args.graph_based](**method_hp_set_copy)
                    preds = train_HGNN_batches(model, data, x, method_hp_set["sample"],
                                               method_hp_set["batch_size"], args.num_workers,
                                               hyperedge_attr, args.patience, args.delta,
                                               weight_true_class, device)
                else:
                    model = HYPERGRAPH_METHODS[args.graph_based](**method_hp_set)
                    preds = train_HGNN(model, data, x, hyperedge_attr, args.patience, args.delta, weight_true_class, device)
            else:
                raise "Unexpected set. Feature method is not set and graph/hypergraph method is unknown."

            # ignore hyperedges predictions
            preds = preds[:data.num_nodes].cpu()
                
                
            local_end = time()

            if int(time())-global_start>args.time_limit:
                break
            # evaluate predictions using Accuracy, ROC-AUC and PR-AUC metrics
            roc_auc, pr_auc = evaluate(preds[data.val_mask], val_y)
            test_roc_auc, test_pr_auc = evaluate(preds[data.test_mask], test_y)
            # save results
            os.makedirs(args.logdir, exist_ok=True) # it is important to create directory not earlier than at least one iteration succeeded
            with open(args.logdir + "/results.csv", "a+") as f:
                embedding_time = embedding_end - local_start
                total_time = local_end - local_start
                f.write(f"{roc_auc},{test_roc_auc},{pr_auc},{test_pr_auc},{embedding_time},{total_time},{embedding_hp_set},\"{method_hp_set}\"\n")
            if pr_auc>best_pr_auc:
                best_pr_auc = pr_auc
                best_hyperparameters = (embedding_hp_set, method_hp_set)
                # save best predictions
                if os.path.isfile(args.logdir+"/preds.pt"):
                    os.remove(args.logdir+"/preds.pt")
                torch.save(preds, args.logdir+"/preds.pt")
        except Exception as e:
            traceback.print_exc()
            print(f"Exception {e.__class__} occured with hyperparameters: {embedding_hp_set} {method_hp_set}")
            if "model" in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
    print(best_hyperparameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys(), type=str,
                        help="Datasets on which hyperparameter tuning should be run.")
    parser.add_argument("--embedding", choices=EMBEDDING_METHODS.keys(), type=str, default=None,
                        help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--feature_based", choices=FEATURE_METHODS.keys(), type=str,
                        help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--graph_based", choices=list(GRAPH_METHODS.keys())+list(HYPERGRAPH_METHODS.keys()),
                        type=str, help="Methods to evaluate during the hyperparameter tuning.")
    parser.add_argument("--graph_repr_embedding", default="incidence", choices=["incidence", "clique"],
                        type=str, help="Graph representation of the hypergraph that should be used for embedding generation." \
                        "Ignored unless 'embedding' contains method that operate on graphs.")
    parser.add_argument("--graph_repr_GNN", default="incidence", choices=["incidence", "clique"],
                        type=str, help="Graph representation of the hypergraph that should be used for GNN methods." \
                        "Ignored unless 'graph_based' contains method that operate on graphs.")
    parser.add_argument("--delta", default=0.001, type=float, help="Only loss change greater than delta"
    "may be counted as positive improvement, otherwise early stopping may be applied.")
    parser.add_argument("--patience", default=5, type=int, help="Maximum number of epochs for which" \
    "loss decrease less than delta may be tollerated.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of parallel workers to use. Use 0 to run everything in the main thread.")
    parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--hyperparam_ranges", default="",
                        type=str, help="Path to the JSON file containing hyperparameter ranges" \
                        "for each method.")
    parser.add_argument("--logdir", type=str, default="logs", help="Default directory for storing logs.")
    parser.add_argument("--time_limit", type=int, default=6000,
                        help="Maximum time in seconds to spend on hyperparameter tuning.")
    parser.add_argument("--train_strategy", type=str, default="none", choices=["none", "weight", \
        "random_oversampling", "random_undersampling", "SMOTE", "tomek_links",
        "hyper_undersampling", "hyperedge_selection"], \
        help="Strategies to improve training process.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if args.hyperparam_ranges=="":
        args.hyperparam_ranges = f"hyperparam_ranges_{args.dataset}.json"
    main(args)
