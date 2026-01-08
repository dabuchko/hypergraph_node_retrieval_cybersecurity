from .hypergraph import Hypergraph
import kagglehub
import torch

class MAWIDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        kaggle_path = "dbuchko/mawi-hypergraph-09042025"
        # load hyperedges
        hyperedge_index = []
        with open(kagglehub.dataset_download(kaggle_path, "hyperedges.csv"), 'r') as f:
            for line in f:
                line = line.strip()
                node, hyperedge = line.split(",")
                hyperedge_index.append((int(node), int(hyperedge)))
        hyperedge_index = torch.tensor(hyperedge_index, dtype=int).T
        # load labels
        labels = []
        with open(kagglehub.dataset_download(kaggle_path, "labels.csv"), 'r') as f:
            for line in f:
                line = line.strip()
                labels.append(int(line))
        labels = torch.tensor(labels, dtype=bool)
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)