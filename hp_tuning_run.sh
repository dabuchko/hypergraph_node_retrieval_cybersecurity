#!/bin/bash

if [ "$1" == "" ]; then
    echo "Dataset is not specified. Please specify dataset as the first argument. \
You may also, optionally, specify sampling strategy as the second argument."
fi

if [ "$2" == "" ]; then
    $2 = "none"
fi

# Feature based
for f in "Logistic Regression" "MLP" "Naive Bayes" "KNN"; do
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Random Gaussian" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Matrix Factorization" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --feature_based="$f" --graph_repr_embedding=clique
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --feature_based="$f" --graph_repr_embedding=clique
done

# General graph based
for f in "GCN", "GIN", "GAT", "GraphSAGE", "Label Propagation"; do
    for repr in "incidence", "clique"; do
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Random Gaussian" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Matrix Factorization" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique
    done
done

# Hypergraph based
for f in "HyperGCN", "UniGCN", "UniGAT", "UniGIN", "UniSAGE", "UniGCNII", "HCHA", "AllDeepSets", "AllSetTransformer", "HNHN", "HGNN", "CSP", "HyperSAGE", "MaxSum"; do
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Random Gaussian" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Matrix Factorization" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_embedding=clique
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$2" --embedding="Node2Vec" --graph_based="$f" --graph_repr_embedding=clique
done
