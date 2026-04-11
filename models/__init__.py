"""
Models module containing implementations of embedding, feature-based, graph-based, and hypergraph models. 
"""

from torch_geometric.nn.models import GCN, GIN, GAT, GraphSAGE, Node2Vec, LabelPropagation
from .embedding.random_gaussian import RandomGaussian
from .embedding.trainable_embeddings_wrapper import trainable_embeddings_wrapper
from .embedding.matrix_factorization import MatrixFactorization
from .embedding.spectral_embedding import SpectralEmbedding, SpectralEmbeddingUnnormalized, SpectralEmbeddingSideNorm, SpectralEmbeddingNorm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from .hypergraph.hypergcn import HyperGCN
from .hypergraph.unignn import UniGCN, UniGAT, UniGIN, UniSAGE, UniGCNII
from .hypergraph.hcha import HCHA
from .hypergraph.allset import AllDeepSets, AllSetTransformer
from .hypergraph.hnhn import HNHN
from .hypergraph.hgnn import HGNN
from .hypergraph.csp import CSP
from .hypergraph.summin import SumMin, SumMinAblation
from .hypergraph.hypersage import HyperSAGE

__all__ = ["EMBEDDING_METHODS", "FEATURE_METHODS", "HYPERGRAPH_METHODS", "GRAPH_METHODS", "RandomGaussian",
           "MatrixFactorization", "SpectralEmbedding", "SpectralEmbeddingUnnormalized", "SpectralEmbeddingSideNorm",
           "SpectralEmbeddingNorm", "Node2Vec", "LogisticRegression", "MLPClassifier", "GaussianNB",
           "KNeighborsClassifier", "HyperGCN", "UniGCN", "UniGAT", "UniGIN", "UniSAGE", "UniGCNII",
           "HCHA", "AllDeepSets", "AllSetTransformer", "HNHN", "HGNN", "CSP", "GCN", "GIN", "GAT",
           "GraphSAGE", "LabelPropagation", "HyperSAGE", "SumMin", "SumMinAblation", "trainable_embeddings_wrapper"]

"""Dictionary mapping embedding methods' names to their corresponding classes."""
EMBEDDING_METHODS = {"Random Gaussian": RandomGaussian, "Trainable Embeddings": trainable_embeddings_wrapper,
                     "Matrix Factorization": MatrixFactorization, "Spectral Embedding": SpectralEmbedding,
                     "Spectral Embedding Unnormalized": SpectralEmbeddingUnnormalized,
                     "Spectral Embedding Side Normalized": SpectralEmbeddingSideNorm,
                     "Spectral Embedding Normalized": SpectralEmbeddingNorm, "Node2Vec": Node2Vec}

"""Dictionary mapping feature methods' names to their corresponding classes."""
FEATURE_METHODS = {"Logistic Regression": LogisticRegression, "MLP": MLPClassifier,
                   "Naive Bayes": GaussianNB, "KNN": KNeighborsClassifier}

"""Dictionary mapping hypergraph methods' names to their corresponding classes."""
HYPERGRAPH_METHODS = {"HyperGCN": HyperGCN, "UniGCN": UniGCN, "UniGAT": UniGAT, "UniGIN": UniGIN,
                      "UniSAGE": UniSAGE, "UniGCNII": UniGCNII, "HCHA": HCHA, "AllDeepSets": AllDeepSets,
                      "AllSetTransformer": AllSetTransformer, "HNHN": HNHN, "HGNN": HGNN, "CSP": CSP,
                      "HyperSAGE": HyperSAGE, "SumMin": SumMin, "SumMinAblation": SumMinAblation}

"""Dictionary mapping graph methods' names to their corresponding classes."""
GRAPH_METHODS = {"GCN": GCN, "GIN": GIN, "GAT": GAT, "GraphSAGE": GraphSAGE, "Label Propagation": LabelPropagation}

