from .hypergraph import Hypergraph
import kagglehub
import torch
import numpy as np

class MH1MDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        path = kagglehub.dataset_download("dbuchko/mh-1m-dataset-hypergraph-representation")
        # load hyperedges
        data = np.loadtxt(path + "/hyperedges.csv", delimiter=",", dtype=np.int32)
        hyperedge_index = torch.from_numpy(data).T
        # load labels
        labels = torch.zeros((hyperedge_index[0].max()+1,), dtype=bool)
        with open(path+"/classes.csv", 'r') as f:
            for line in f:
                line = line.strip()
                labels[int(line)] = True
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
