#!/usr/bin/env python3
import torch
from data import *
import argparse
import networkx as nx
import igraph

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=DATASETS.keys(), help="Hypergraph dataset " \
                    "on which statistics should be computed. Hypergraph dataset should be a child of " \
                    "data.Hypergraph class and be distributed by data module-folder.")
parser.add_argument("--seed", default=42, type=int, help="Random seed for datasets initializations (useful if random splits are used).")
parser.add_argument("--arguments", default={}, type=dict, help="Dictionary of dataset arguments.")

def main(dataset: Hypergraph) -> None:
    # print straightforward properties: node number, number of edges,
    # malicious node number, benign node number
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of edges: {dataset.num_edges}")
    print(f"Number of incident connections: {dataset.hyperedge_index.shape[1]}")
    if dataset.y != None:
        mal_num = (dataset.y==1).sum()
        ben_num = (dataset.y==0).sum()
        print(f"Number of malicious nodes: {mal_num} ({mal_num/dataset.num_nodes*100:.4f}%)")
        print(f"Number of benign nodes: {ben_num} ({ben_num/dataset.num_nodes*100:.4f}%)")

    # compute and print the number of empty and simple edges by iterating over
    # hyperedges indices
    empty_edges = set(range(dataset.num_edges))
    simple_edges = set()
    for i in range(dataset.hyperedge_index.shape[1]):
        hyperedge = int(dataset.hyperedge_index[1, i].item())
        if hyperedge in empty_edges:
            empty_edges.remove(hyperedge)
            simple_edges.add(hyperedge)
        elif hyperedge in simple_edges:
            simple_edges.remove(hyperedge)
    print(f"Number of empty hyperedges: {len(empty_edges)}")
    print(f"Number of simple hyperedges: {len(simple_edges)}")
    del empty_edges
    
    # compute the number of isolated nodes. Isolated are all nodes that do not
    # belong to some non-simple hyperedge
    isolated_nodes = set(range(dataset.num_nodes))
    for i in range(dataset.hyperedge_index.shape[1]):
        node = int(dataset.hyperedge_index[0, i].item())
        if node not in isolated_nodes:
            continue
        hyperedge = int(dataset.hyperedge_index[1, i].item())
        if hyperedge not in simple_edges:
            isolated_nodes.remove(node)

    print(f"Number of isolated nodes: {len(isolated_nodes)}")

    # compute hypernode degree, hyperedge degree, and hyperedge weight statistics
    node_degrees = torch.bincount(dataset.hyperedge_index[0]).float()
    print(f"Hypernode degree: median {node_degrees.median()}, mean {node_degrees.mean()}, standard deviation {node_degrees.std()}, maximum {node_degrees.max()}, minimum {node_degrees.min()}")
    hyperedge_degrees = torch.bincount(dataset.hyperedge_index[1]).float()
    print(f"Hyperedge degree: median {hyperedge_degrees.median()}, mean {hyperedge_degrees.mean()}, standard deviation {hyperedge_degrees.std()}, maximum {hyperedge_degrees.max()}, minimum {hyperedge_degrees.min()}")
    if dataset.hyperedge_weight is not None:
        he_float_weight = dataset.hyperedge_weight.float()
        print(f"Hyperedge weight: median {he_float_weight.median()}, mean {he_float_weight.mean()}, standard deviation {he_float_weight.std()}, maximum {he_float_weight.max()}, minimum {he_float_weight.min()}")
    
    # compute what fraction of incidence matrix is filled with entry bigger than 0
    print(f"Hypergraph density: {dataset.hyperedge_index.shape[1] * 100 / (dataset.num_nodes * dataset.num_edges):.4f}%")

    # count number of the connected components
    hypernode_labels = torch.arange(dataset.num_nodes)
    hyperedge_labels = torch.arange(dataset.num_nodes, dataset.num_nodes+dataset.num_edges)
    hypernode_labels_dict = {i: {i} for i in range(dataset.num_nodes)}
    hyperedge_labels_dict = {dataset.num_nodes+i: {i} for i in range(dataset.num_edges)}
    for i in range(dataset.hyperedge_index.shape[1]):
        hypernode = dataset.hyperedge_index[0, i].item()
        hyperedge = dataset.hyperedge_index[1, i].item()
        hypernode_label = hypernode_labels[hypernode].item()
        hyperedge_label = hyperedge_labels[hyperedge].item()
        if hypernode_label < hyperedge_label:
            for n in hyperedge_labels_dict[hyperedge_label]:
                hyperedge_labels[n] = hypernode_label
            if hypernode_label not in hyperedge_labels_dict:
                hyperedge_labels_dict[hypernode_label] = set()
            hyperedge_labels_dict[hypernode_label] = hyperedge_labels_dict[hypernode_label].union(hyperedge_labels_dict[hyperedge_label])
            del hyperedge_labels_dict[hyperedge_label]
        elif hypernode_label > hyperedge_label:
            for n in hypernode_labels_dict[hypernode_label]:
                hypernode_labels[n] = hyperedge_label
            if hyperedge_label not in hypernode_labels_dict:
                hypernode_labels_dict[hyperedge_label] = set()
            hypernode_labels_dict[hyperedge_label] = hypernode_labels_dict[hyperedge_label].union(hypernode_labels_dict[hypernode_label])
            del hypernode_labels_dict[hypernode_label]

    num_of_conn_comp = torch.unique(hypernode_labels).shape[0]
    print(f"Number of connected components: {num_of_conn_comp}")

    # compute statistics of number of hypernodes across components
    components_stat = torch.bincount(hypernode_labels).float()
    print(f"Number of hypernodes in component: median {components_stat.median()}, mean {components_stat.mean()}, standard deviation {components_stat.std()}, maximum {components_stat.max()}, minimum {components_stat.min()}")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    print(f"Analyzing dataset {args.dataset}")
    main(DATASETS[args.dataset](**args.arguments))