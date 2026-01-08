from .hypergraph import Hypergraph
import torch

class CiscoEmailDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        raise Exception("CiscoEmail is a private dataset, currently unavailable publicly.")