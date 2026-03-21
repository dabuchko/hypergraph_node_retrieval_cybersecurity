#!/bin/bash

if [ "$1" == "" ]; then
    echo "Dataset is not specified. Please specify dataset as the first argument. \
You may also, optionally, specify sampling strategy as the second argument."
fi

imbalance=${2:-none}

# Feature based

for f in "Logistic Regression" "MLP" "Naive Bayes" "KNN"; do
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Random Gaussian" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Matrix Factorization" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --feature_based="$f" --graph_repr_embedding=clique
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --feature_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --feature_based="$f" --graph_repr_embedding=clique
done

# General graph based

for repr in "incidence" "clique"; do
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --graph_based="Label Propagation" --graph_repr_GNN="$repr"
done

for f in "GCN" "GIN" "GAT" "GraphSAGE"; do
    for repr in "incidence" "clique"; do
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Random Gaussian" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Matrix Factorization" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr"
        python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique
    done
done

# Hypergraph based

python hyperparam_tuning.py "$1" --imbalance="$imbalance" --graph_based="CSP"

for f in "HyperGCN" "UniGCN" "UniGAT" "UniGIN" "UniSAGE" "UniGCNII" "HCHA" "AllDeepSets" "AllSetTransformer" "HNHN" "HGNN" "HyperSAGE" "MaxSum"; do
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Random Gaussian" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Matrix Factorization" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_embedding=clique
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --graph_based="$f"
    python hyperparam_tuning.py "$1" --imbalance="$imbalance" --embedding="Node2Vec" --graph_based="$f" --graph_repr_embedding=clique
done
