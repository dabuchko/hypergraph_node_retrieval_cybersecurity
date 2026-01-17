#!/usr/bin/env python3
import torch
from data import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=DATASETS.keys(), required=True, help="Hypergraph dataset " \
                    "on which statistics should be computed. Hypergraph dataset should be a child of " \
                    "data.Hypergraph class and be distributed by data module-folder.")
parser.add_argument("--seed", default=42, type=int, help="Random seed for datasets initializations (useful if random splits are used).")
parser.add_argument("--arguments", default={}, type=dict, help="Dictionary of ")

def main(dataset: Hypergraph) -> None:
    # print straightforward properties: node number, number of edges,
    # malicious node number, benign node number
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of edges: {dataset.num_edges}")
    if dataset.y != None:
        print(f"Number of malicious nodes: {(dataset.y==1).sum()}")
        print(f"Number of benign nodes: {(dataset.y==0).sum()}")

    # compute and print the number of empty and simple edges by iterating over
    # hyperedges matrix
    empty_edges = set(range(dataset.num_edges))
    simple_edges = set()
    for i in range(dataset.hypergraph_index.shape[1]):
        hyperedge = int(dataset.hypergraph_index[1, i].item())
        if hyperedge in empty_edges:
            empty_edges.remove(hyperedge)
            simple_edges.add(hyperedge)
        elif hyperedge in simple_edges:
            simple_edges.remove(hyperedge)
    print(f"Number of empty edges: {len(empty_edges)}")
    print(f"Number of simple edges: {len(simple_edges)}")
    
    # compute the number of isolated nodes. Isolated are all nodes that do not
    # belong to some non-simple hyperedge
    isolated_nodes = set(range(dataset.num_nodes))
    for i in range(dataset.hypergraph_index.shape[1]):
        node = int(dataset.hypergraph_index[0, i].item())
        hyperedge = int(dataset.hypergraph_index[1, i].item())
        if hyperedge not in simple_edges:
            isolated_nodes.remove(node)

    print(f"Number of isolated nodes: {len(isolated_nodes)}")

    # TODO: Modularity,
    # Average path length,
    # Network diameter,
    # Clustering coefficient,
    # Connect threshold,
    # Connected components

    node_degrees = torch.bincount(dataset.hypergraph_index[0])
    print(f"Node degree: median {node_degrees.median()}, mean {node_degrees.mean()}, standard deviation {node_degrees.std()}")
    hyperedge_degrees = torch.bincount(dataset.hypergraph_index[1])
    print(f"Edge degree: median {hyperedge_degrees.median()}, mean {hyperedge_degrees.mean()}, standard deviation {hyperedge_degrees.std()}")
    if dataset.hyperedge_weight!=None:
        print(f"Edge weight: median {dataset.hyperedge_weight.median()}, mean {dataset.hyperedge_weight.mean()}, standard deviation {dataset.hyperedge_weight.std()}")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    print(f"Analyzing dataset {args.dataset}")
    main(DATASETS[args.dataset](**args.arguments))