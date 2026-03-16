from .hypergraph import Hypergraph
import kagglehub
import torch
import numpy as np

class MAWIDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        kaggle_path = "dbuchko/mawi-hypergraph-09042025"
        # load hyperedges
        hyperedge_index = np.loadtxt(
            kagglehub.dataset_download(kaggle_path, "hyperedges.csv"),
            delimiter=",",
            dtype=np.int64
        )
        hyperedge_index = torch.from_numpy(hyperedge_index).T
        # load labels
        labels = torch.from_numpy(
            np.loadtxt(kagglehub.dataset_download(kaggle_path, "labels.csv"), dtype=np.bool_)
        )
        labels = torch.from_numpy(labels)
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)