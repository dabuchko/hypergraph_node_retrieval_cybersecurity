from torch_geometric.nn.models import GCN, GIN, GAT, GraphSAGE, Node2Vec, LabelPropagation
from .embedding.random_gaussian import RandomGaussian
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

__all__ = ["EMBEDDING_METHODS", "FEATURE_METHODS", "HYPERGRAPH_METHODS", "GRAPH_METHODS", "RandomGaussian",
           "MatrixFactorization", "SpectralEmbedding", "SpectralEmbeddingUnnormalized", "SpectralEmbeddingSideNorm",
           "SpectralEmbeddingNorm", "Node2Vec", "LogisticRegression", "GaussianNB"]

EMBEDDING_METHODS = {"Random Gaussian": RandomGaussian, "Matrix Factorization": MatrixFactorization,
                     "Spectral Embedding": SpectralEmbedding,
                     "Spectral Embedding Unnormalized": SpectralEmbeddingUnnormalized,
                     "Spectral Embedding Side Normalized": SpectralEmbeddingSideNorm,
                     "Spectral Embedding Normalized": SpectralEmbeddingNorm, "Node2Vec": Node2Vec}
FEATURE_METHODS = {"Logistic Regression": LogisticRegression, "MLP": MLPClassifier,
                   "Naive Bayes": GaussianNB, "KNN": KNeighborsClassifier}
HYPERGRAPH_METHODS = {"HyperGCN": HyperGCN, "UniGCN": UniGCN, "UniGAT": UniGAT, "UniGIN": UniGIN,
                      "UniSAGE": UniSAGE, "UniGCNII": UniGCNII, "HCHA": HCHA, "AllDeepSets": AllDeepSets,
                      "AllSetTransformer": AllSetTransformer, "HNHN": HNHN, "HGNN": HGNN, "CSP": CSP}
GRAPH_METHODS = {"GCN": GCN, "GIN": GIN, "GAT": GAT, "GraphSAGE": GraphSAGE, "LabelPropagation": LabelPropagation}

