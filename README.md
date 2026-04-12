# Retrieval of hypernodes in high-volume hypergraph in cybersecurity

> The content of this repository was developed as part of the bachelor's thesis
"Retrieving Rare Entities of Interest from High-Volume Hypergraph Data in Cybersecurity".


The current repository tests multiple methods for hypernode retrieval
in hypergraphs in cybersecurity. Hypernodes and hyperedges in hypergraphs do not
have any features, so training and inference should be based solely on the relational
structure. The data is usually imbalanced, e.g. malicious entities are highly
underrepresented and majority of entities belong to benign class. For this reason
primarily PR AUC (Average Precision) metric is used for evaluation and multiple
imbalance handling strategies are adopted.

The following datasets were used in the experiments:

* Private **CiscoEmail** dataset - this is a private hypergraph email dataset
extracted from the emails of Cisco Systems Inc. clients.
It is not publicly distributed, and it is not included in the current repository.
The dataset contains two types of emails as hypernodes (benign and malicious).
Malicious emails include spam and phishing emails. Hyperedges represent shared
attributes of emails, such as hyperlinks, attachments, etc. Hypernode belongs to
hyperedge if the email represented by this hypernode includes attribute represented by
this hyperedge.

* **SpamAssassin** hypergraph dataset is a hypergraph representation of SpamAssassin dataset.
Hypernodes represent email of two types, malicious (spam or phishing) and benign.
Hyperedges represent word n-grams, hypernode belongs to hyperedge if the email
represented by the corresponding hypernode has the corresponding n-gram in its text.
Hyperedge weights are assigned based on the number of grams represented by the corresponding
hyperedge.

Original dataset source: https://huggingface.co/datasets/talby/spamassassin

* Hypergraph representation of **BCCC-VulSCs-2023** dataset consists of 36,670 Solidity Smart Contracts.
Each hypernode represents a source code file of the contract, and each hyperedge represents an n-gram
extracted from the source codes. Hypernodes of two types are present: vulnerable and not vulnerable.
Hyperedge weight is defined as the number of tokens in the n-gram. Since by definition
hypergraph cannot contain duplicated hyperedges, similar hyperedges were merged together
by summing their weights.

Original dataset source: https://www.kaggle.com/datasets/bcccdatasets/bccc-vulscs-2023

* Hypergraph representation of **MH-1M** dataset consists of Android applications that are
classified among two classes: malicious and benign. In this hypergraph, hypernodes
represent Android applications, and hyperedges represent their shared attributes
(Android API calls, Android Intents, Android permissions, Opcodes). Hypernode belongs
to hyperedge if the Android application represented by the corresponding hypernode
has a shared attribute represented by the corresponding hyperedge.

Original dataset source: https://github.com/Malware-Hunter/MH-1M

Hypergraph representation source: https://www.kaggle.com/datasets/dbuchko/mh-1m-dataset-hypergraph-representation

* Hypergraph representation of **MAWI** dataset consists of 24 hour trace at the
transit link of Widely Integrated Distributed Environment (WIDE) to the upstream ISP.
In constructed hypergraph representation, hypernodes represent hosts in WIDE network,
and hyperedges represent hosts outside the WIDE network. Hypernode belongs to hyperedge
if hosts represented by the corresponding hypernode and hyperedge have ever established
a connection during 24 hour captured trace. Each hypernode belongs to either of two
classes: malicious host or benign host. Malicious hosts are identified based on the
public lists of malicious ip addresses.

Original dataset source: https://mawi.wide.ad.jp/mawi/ditl/ditl2025/

Hypergraph representation source: https://www.kaggle.com/datasets/dbuchko/mawi-hypergraph-09042025

The experiments conducted on these datasets are described further in this document in individual sections.
SpamAssassin dataset was excluded from the experiments because trivial methods can easily
achieve Average Precision of more than 99%, this makes any comparison between methods impossible.

## Prerequisites

Prior to running the code the libraries specified in the [requirements.txt](requirements.txt)
should be satisfied. It is recommended to proceed as follows:

1. Install PyTorch by following the [official guide](https://pytorch.org/get-started/locally/).

2. Install PyTorch Geometric framework with all optional dependencies by following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

3. Install the rest of the required libraries automatically by running the following command
(if "venv" virtual environment is used):

```
pip install -r requirements.txt
```

## Dataset analysis

All datasets used during the experiments are analyzed for different metrics: 

* Number of hypernodes
* Number of hyperedges
* Number of incident connections
* Number of malicious and benign nodes
* Number of empty hyperedges
* Number of simple hyperedges (where hyperedge degree is 1)
* Number of isolated nodes
* Hypernode degree statistics (mean, median, standard deviation, minimum, maximum)
* Hyperedge degree statistics (mean, median, standard deviation, minimum, maximum)
* Hyperedge weight statistics (mean, median, standard deviation, minimum, maximum)
* Hypergraph density - fraction of incidence connections from the maximum
possible number of connections
* Number of connected components
* Statistics on the number of hypernodes in connected component (mean, median, standard deviation, minimum, maximum)

Statistics of the dataset can be analyzed by using the following command, where
`<DATASET>` should be replaced by the actual dataset name:

```
python stats.py <DATASET>
```

## Hyperparameter tuning

Hyperparameter tuning finds the best set of hyperparameters for selected
method and dataset using Random Search method. The hyperparameter tuning can
be started for feature-based method using the following command, where
`<DATASET>` stands for dataset name, `<METHOD>` stands for the name of the
feature based method, and `<EMBEDDING>` for embedding method name:

```
python hyperparam_tuning.py <DATASET> --feature_based=<METHOD> --embedding=<EMBEDDING>
```

Hyperparameter tuning for graph-based and hypergraph-based methods can be strated
using the following command, where `<DATASET>` stands for dataset name, `<METHOD>`
stands for the name of the used method, and `<EMBEDDING>` for embedding method name:

```
python hyperparam_tuning.py <DATASET> --graph_based=<METHOD> --embedding=<EMBEDDING>
```

Multiple strategies to improve training are available by using "--train_strategy"
argument, almost all of the available strategies are designed to handle class imbalance,
except "hyperedge_selection" that does not directly contribute to alleviating class
imbalance issue, still it decreases the amount of consumed resources during training,
and may contribute for improving precision.

To specify the graph representation of the hypergraph that should be used to
train graph-based methods, the "--graph_repr_GNN" argument should be used.
Only two representations are available: incidence representation and
clique representation. If no representation is specified, by default incidence
representation is used.

Graph representation can also be set for some embedding methods that work
on graphs, like Spectral Embedding or Node2Vec. In this case "--graph_repr_embedding"
should be used. Similarly, only two representations are available: incidence representation and
clique representation, and if no representation is specified, by default incidence
representation is used.

To specify path to the JSON file containing ranges of hyperparameters the
"--hyperparam_ranges" argument should be used. If no path is specified the
`hyperparam_ranges_<DATASET>.json` is used instead, where `<DATASET>` is
replaced with dataset name. The format of JSON file is the following:
it should be a dictionary, where keys are method names and values are dictionaries
with hyperparameter names as keys and list of hyperparameter values as values
in these dictionaries. The example of such file can be found in root of this repository.

More detailed information about the available arguments and their values can be
found by executing the following command:

```
python hyperparam_tuning.py --help
```

To run hyperparameter tuning for all available methods, the following command
can be used, where `<DATASET>` should be replaced with actual dataset name,
`<TRAIN_STRATEGY>` stands for training strategy, next additional arguments
may follow that will be passed in addition to every "hyperparam_tuning.py"
call in script:

```
./hp_tuning_run.sh <DATASET> <TRAIN_STRATEGY> <ADDITIONAL_ARGUMENTS>
```

To run hyperparameter tuning for ablations of the methods use another script,
where command arguments have the same meaning as above:

```
./hp_tuning_abl_run.sh <DATASET> <TRAIN_STRATEGY> <ADDITIONAL_ARGUMENTS>
```