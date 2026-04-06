from .hypergraph import Hypergraph
import kagglehub
import torch
import numpy as np

class MAWIDataset(Hypergraph):
    """
    Source: https://www.kaggle.com/datasets/dbuchko/mawi-hypergraph-09042025

    This is a preprocessed version of network connections from the traffic dataset
    collected by the MAWI Working Group of the WIDE Project. The original dataset
    includes traffic data from samplepoint-F that was collected for 24 hours 9th April 2025
    at the transit link of WIDE to the upstream ISP. More information about the original
    dataset is available here: https://mawi.wide.ad.jp/mawi/

    Based on the original data, the current hypergraph version was created.
    In this version hypernodes represent hosts located outside Japan, hyperedges
    represent hosts of the WIDE network. The given hypernode belongs to hyperedge
    if the hosts represented by hypernode and hyperedge have ever exchanged packets
    during the 24 hour trace at 9th April 2025. Each hypernode received a label,
    hypernodes that represent malicious hosts received label 1, and hypernodes
    that represent benign hosts received label 0.
    """
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2):
        """
        Initializes a hypergraph representation of the MAWI dataset.

        :param train_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the training set.
        :type train_size: float
        :param val_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the validation set.
        :type val_size: float
        """
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
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)