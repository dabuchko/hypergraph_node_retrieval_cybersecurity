#!/bin/bash

if [ "$1" == "" ]; then
    echo "Dataset is not specified. Please specify dataset as the first argument. \
You may also, optionally, specify sampling strategy as the second argument."
fi

train_strategy=${2:-none}

# Feature based

for f in "Logistic Regression" "MLP" "Naive Bayes" "KNN"; do
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Matrix Factorization" --feature_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --feature_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --feature_based="$f" --graph_repr_embedding=clique "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --feature_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --feature_based="$f" --graph_repr_embedding=clique "${@:3}"
done

# General graph based

for repr in "incidence" "clique"; do
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --graph_based="Label Propagation" --graph_repr_GNN="$repr" "${@:3}"
done

for f in "GCN" "GIN" "GAT" "GraphSAGE"; do
    for repr in "incidence" "clique"; do
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Random Gaussian" --graph_based="$f" --graph_repr_GNN="$repr" "${@:3}"
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Matrix Factorization" --graph_based="$f" --graph_repr_GNN="$repr" "${@:3}"
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr" "${@:3}"
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique "${@:3}"
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr" "${@:3}"
        python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --graph_based="$f" --graph_repr_GNN="$repr" --graph_repr_embedding=clique "${@:3}"
    done
done

# Hypergraph based

python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --graph_based="CSP" "${@:3}"
python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --graph_based="SumMin" "${@:3}"

for f in "HyperGCN" "UniGCN" "UniGAT" "UniGIN" "UniSAGE" "UniGCNII" "HCHA" "AllDeepSets" "AllSetTransformer" "HNHN" "HGNN" "HyperSAGE"; do
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Random Gaussian" --graph_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Matrix Factorization" --graph_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --graph_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Spectral Embedding" --graph_based="$f" --graph_repr_embedding=clique "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --graph_based="$f" "${@:3}"
    python hyperparam_tuning.py "$1" --train_strategy="$train_strategy" --embedding="Node2Vec" --graph_based="$f" --graph_repr_embedding=clique "${@:3}"
done
