# Developer Documentation

This file provides a brief overview of the codebase and its structure.
More detailed documentation of classes, functions, and their arguments
can be found in their corresponding Python docstrings, if available.

* `requirements.txt` - list of the required Python libraries to run the experiments.
* `stats.py` - script for analyzing the dataset statistics.
* `hyperparam_tuning.py` - script for hyperparameter tuning of the considered methods on the specified datasets.
* `train.py` - script for training the models. Its functions are primarily used
    in `hyperparam_tuning.py` script.
    - `train_predict_node2vec` - function for training and evaluating node2vec method.
    - `train_GNN` - function for training and evaluating trainable graph-based methods as whole (not in batches).
    - `train_GNN_batches` - function for training and evaluating trainable graph-based methods in batches.
    - `validate_and_preprocess_graph` - function for validating and preprocessing the graph data before training.
    - `train_HGNN` - function for training and evaluating trainable hypergraph-based methods as whole (not in batches).
    - `train_HGNN_batches` - function for training and evaluating trainable hypergraph-based methods in batches.
    -  `HGNN_batch_sampler` - batch sampler class for dataloader used for training hypergraph-based methods in batches.
    - `validate_and_preprocess_hypergraph` - function for validating and preprocessing the hypergraph data before training.
    - `evaluate` - function for evaluating the generated predictions using ROC-AUC and PR-AUC (Average Precision) metrics.
* `hp_tuning_run.sh` - script for running hyperparameter tuning for all available methods on the specified dataset with specified training strategy and additional arguments.
* `hp_tuning_abl_run.sh` - script for running hyperparameter tuning for ablations of the methods on the specified dataset with specified training strategy and additional arguments.
* `hyperparam_ranges_<DATASET>.json` - JSON file containing ranges of hyperparameters for the specified dataset. The format of JSON file is the following: it should be a dictionary, where keys are method names and values are dictionaries with hyperparameter names as keys and list of hyperparameter values as values in these dictionaries.
* `hypergraph_undersampling.py` - script containing two techniques for undersampling the hypergraph data described in "Hypergraph-based importance assessment for binary classification data" paper: "hyper_undersampling" and "hyperedge_selection".
These techniques were modified to use the exact computation of the average of labels of the neighborhood of the hypernodes instead of the approximation using the random walks on the hypergraph.
    - `hyper_undersampling` - function for undersampling the hypernodes in the hypergraph data.
    - `hyperedge_selection` - function for selecting the most important hyperedges in the hypergraph data.
* `data/` - directory containing classes for dataset loading and preprocessing. More detailed information about the datasets can be found in the [README.md](./README.md) file.
    - `hypergraph.py`
        - `Hypergraph` - a base class for hypergraph dataset loading and preprocessing.
    - `CiscoEmail.py`
        - `CiscoEmailDataset` - class for loading and preprocessing CiscoEmail dataset, inherits from `hypergraph.py`. Since CiscoEmail dataset is a private dataset, the exact code used during the experiments presented in the thesis is excluded from the repository.
    - `BCCCVulSCs2023.py`
        - `BCCCVulSCs2023Dataset` - class for loading and preprocessing BCCCVulSCs2023 dataset, inherits from `hypergraph.py`.
    - `SpamAssassin.py`
        - `SpamAssassinDataset` - class for loading and preprocessing SpamAssassin dataset, inherits from `hypergraph.py`. Due to its simplicity, this dataset is not used in the experiments presented in the thesis.
    - `MAWI.py`
        - `MAWIDataset` - class for loading and preprocessing MAWI dataset, inherits from `hypergraph.py`.
    - `MH_1M.py`
        - `MH1MDataset` - class for loading and preprocessing MH_1M dataset, inherits from `hypergraph.py`.
* `models/` - directory containing all the implemented methods used for experiments in the thesis. The methods are organized in categories: embedding generation methods, feature-based methods, graph-based methods, and hypergraph-based methods.
    - `__init__.py` - this file does not only contain the imports of all the implemented methods, but also imports other methods from the PyTorch Geometric and Scikit-learn libraries. All feature-based methods are imported from Scikit-learn, and almost all graph-based methods are imported from PyTorch Geometric, except for Label Propagation method which PyTorch Geometric implementation was memory inefficient.
    - `embedding/` - directory containing embedding generation methods.
        - `random_uniform.py`
            - `XavierUniform` - class for generating random uniform embeddings using Xavier initialization.
        - `random_gaussian.py`
            - `RandomGaussian` - class for generating random Gaussian embeddings.
        - `matrix_factorization.py`
            - `MatrixFactorization` - class for generating embeddings using matrix factorization method.
        - `spectral_embedding.py`
            - `_construct_unnormalized_laplacian` - function for constructing unnormalized Laplacian matrix from the graph data.
            - `SpectralEmbeddingUnnormalized` - class for generating embeddings using unnormalized spectral embedding method.
            - `SpectralEmbeddingSideNorm` - class for generating embeddings using side-normalized spectral embedding method.
            - `SpectralEmbeddingNorm` - class for generating embeddings using symmetrically normalized spectral embedding method.
            - `SpectralEmbedding` - the central class for generating spectral embeddings, it allows to choose the spectral embedding algorithm from the three algorithms listed above.
        - `trainable_embeddings_wrapper.py`
            - `trainable_embeddings_wrapper` - function that generates a trainable embedding matrix as PyTorch parameter.
    - `graph/` - directory containing graph-based methods.
        - `label_propagation.py`
            - `LabelPropagation` - class that implements the label propagation method.
    - `hypergraph/` - directory containing hypergraph-based methods.
        - `hyperedge_aggr.py`
            - `HyperedgeAggregation` - class that implements the aggregation of hypernode features to obtain hyperedge features representation. This class is used because MessagePassing class from PyTorch Geometric library allows only for one type of aggregation. This class is a flexible way for the methods to instantiate another MessagePassing layer.
        - `basic_hgnn.py`
            - `BasicHGNN` - class that implements the basic hypergraph neural network method based
            on the provided hypergraph layer.
        - `allset.py`
            - `AllDeepSets` - class that implements the AllDeepSets method proposed in "You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks" paper.
            - `AllSetTransformer` - class that implements the AllSetTransformer method proposed in "You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks" paper.
            - all the remaining classes in this file are adaptations of functions and classes presented in: https://github.com/jianhao2016/AllSet/tree/main
        - `csp.py`
            - `CSP` - class that implements the CSP method proposed in "CSP: An Efficient Baseline for Learning on Large-Scale Structured Data" paper.
        - `hcha.py`
            - `HCHA` - class that implements the HCHA method proposed in "Hypergraph Convolution and Hypergraph Attention" paper.
        - `hgnn.py`
            - `HGNN` - class that implements the HGNN method proposed in "Hypergraph Neural Networks" paper.
        - `hnhn.py`
            - `HNHN` - class that implements the HNHN method proposed in "HNHN: Hypergraph Networks with Hyperedge Neurons" paper.
            - `HNHNConv` - class that implements the layer of HNHN method.
        - `hypergcn.py`
            - `HyperGCN` - class that implements the HyperGCN method proposed in "HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs" paper.
            - `HyperGCNConv` - class that implements the layer of HyperGCN method.
        - `hypersage.py`
            - `HyperSAGE` - class that implements the HyperSAGE method proposed in "HyperSAGE: Generalizing Inductive Representation Learning on Hypergraphs" paper.
            - `HyperSAGEConv` - class that implements the layer of HyperSAGE method.
        - `unignn.py` - implements all methods described in "UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks" paper and their corresponding layers.
            - `UniGCN` - class that implements the UniGCN method from UniGNN paper.
            - `UniGCNConv` - class that implements the layer of the UniGCN method.
            - `UniGAT` - class that implements the UniGAT method from UniGNN paper.
            - `UniGATConv` - class that implements the layer of the UniGAT method.
            - `UniGIN` - class that implements the UniGIN method from UniGNN paper.
            - `UniGINConv` - class that implements the layer of the UniGIN method.
            - `UniSAGE` - class that implements the UniSAGE method from UniGNN paper.
            - `UniSAGEConv` - class that implements the layer of the UniSAGE method.
            - `UniGCNII` - class that implements the UniGCNII method from UniGNN paper.
            - `UniGCNIIConv` - class that implements the layer of the UniGCNII method.
        - `summin.py`
            - `SumMin` - class that implements the SumMin method proposed in the thesis.
            - `SumMinConv` - class that implements the layer of the SumMin method.
            - `SumMinAblation` - class that implements the ablation of the SumMin method that accepts
                hypernode embeddings instead of using the trainable hypernode embeddings.