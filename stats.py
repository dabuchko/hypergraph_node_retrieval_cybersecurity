#!/usr/bin/env python3
import torch
from data import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=DATASETS.keys(), help="Hypergraph dataset " \
                    "on which statistics should be computed. Hypergraph dataset should be a child of " \
                    "data.Hypergraph class and be distributed by data module-folder.")
parser.add_argument("--seed", default=42, type=int, help="Random seed for datasets initializations (useful if random splits are used).")
parser.add_argument("--arguments", default={}, type=dict, help="Dictionary of dataset arguments.")

def main(dataset: Hypergraph) -> None:
    # print straightforward properties: node number, number of edges,
    # malicious node number, benign node number

    # count number of the connected components
    hyperedge_index = dataset.hyperedge_index[:, torch.argsort(dataset.hyperedge_index[0])]
    hyperedge_index = hyperedge_index[:, torch.argsort(hyperedge_index[1])]
    hypernode_labels = torch.arange(dataset.num_nodes)
    last_hyperedge = hyperedge_index[1, 0].item()
    hypernode_start_index = 0
    for i in range(hyperedge_index.shape[1]):
        hyperedge = hyperedge_index[1, i].item()
        if hyperedge!=last_hyperedge:
            hypernode_labels[hyperedge_index[0, hypernode_start_index:i]] = hypernode_labels[hyperedge_index[0, hypernode_start_index:i]].min()
            last_hyperedge = hyperedge
            hypernode_start_index = i

    num_of_conn_comp = torch.unique(hypernode_labels).shape[0]
    print(f"Number of connected components: {num_of_conn_comp}")

    # compute statistics of number of hypernodes across components
    hypernode_labels_values = hypernode_labels.unique()
    hypernode_new_indexes = torch.empty((dataset.num_nodes,), dtype=torch.long) # reindex labels to avoid components of 0 size
    hypernode_new_indexes[hypernode_labels_values] = torch.arange(hypernode_labels_values.shape[0])
    hypernode_labels = hypernode_new_indexes[hypernode_labels]
    components_stat = torch.bincount(hypernode_labels).float()
    print(f"Number of hypernodes in component: median {components_stat.median()}, mean {components_stat.mean()}, standard deviation {components_stat.std()}, maximum {components_stat.max()}, minimum {components_stat.min()}")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    print(f"Analyzing dataset {args.dataset}")
    main(DATASETS[args.dataset](**args.arguments))