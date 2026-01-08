from .hypergraph import Hypergraph
import kagglehub
import torch

class MH1MDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2):
        path = kagglehub.dataset_download("dbuchko/mh-1m-dataset-hypergraph-representation")
        # load hyperedges
        hyperedge_index = []
        with open(path+"/hyperedges.csv", 'r') as f:
            for line in f:
                line = line.strip()
                node, hyperedge = line.split(',')
                hyperedge_index.append((int(node), int(hyperedge)))
        hyperedge_index = torch.tensor(hyperedge_index, dtype=int).T
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
    