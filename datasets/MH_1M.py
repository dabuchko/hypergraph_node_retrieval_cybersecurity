from .hypergraph import Hypergraph
import kagglehub
import torch

class MH1MDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        path = kagglehub.dataset_download("dbuchko/mh-1m-dataset-hypergraph-representation")
        # load hyperedges
        hyperedges = []
        with open(path+"/hyperedges.csv", 'r') as f:
            for line in f:
                line = line.strip()
                node, hyperedge = line.split()
                hyperedges.append((int(node), int(hyperedge)))
        hyperedges = torch.tensor(hyperedges, dtype=int).T
        # load labels
        labels = []
        with open(path+"/labels.csv", 'r') as f:
            for line in f:
                line = line.strip()
                labels.append(int(line))
        labels = torch.tensor(labels, dtype=bool)
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedges, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    