from .hypergraph import Hypergraph
import kagglehub
import torch
import numpy as np

class MH1MDataset(Hypergraph):
    """
    Source: https://www.kaggle.com/datasets/dbuchko/mh-1m-dataset-hypergraph-representation

    This is a preprocessed version of MH-1M dataset converted into the hypergraph representation.
    Hypernodes represent Android applications, and hyperedges represent shared attributes of applications
    (Android API calls, Android Intents, Android permissions, Opcodes). Hypernode label is set to 1
    if the corresponding application is malicious and 0 otherwise.
    """
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2):
        """
        Initializes a hypergraph representation of the MH-1M dataset.

        :param train_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the training set.
        :type train_size: float
        :param val_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the validation set.
        :type val_size: float
        """
        path = kagglehub.dataset_download("dbuchko/mh-1m-dataset-hypergraph-representation")
        # load hyperedges
        data = np.loadtxt(path + "/hyperedges.csv", delimiter=",", dtype=np.int64)
        hyperedge_index = torch.from_numpy(data).T
        # load labels
        labels = torch.from_numpy(
            np.loadtxt(path + "/classes.csv", dtype=np.bool_)
        )
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
