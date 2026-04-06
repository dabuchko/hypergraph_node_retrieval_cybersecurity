from .hypergraph import Hypergraph
import kagglehub
import torch
import os
from sklearn.feature_extraction.text import CountVectorizer
import random

class BCCCVulSCs2023Dataset(Hypergraph):
    """
    Dataset class for constrcuting a hypergraph representation of the BCCC-VulSCs-2023 dataset.
    Each hypernode represents a source code file, and each hyperedge represents an n-gram,
    where n-gram range is provided in initialization argument. Hypernode label is 1 if
    the source code file contains a vulnerability and 0 otherwise.
    Hyperedge weight is defined as the number of tokens in the n-gram. Since by definition
    hypergraph cannot contain duplicated hyperedges, similar hyperedges were merged
    by summing their weights.
    """
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2, ngram_range: tuple[int, int] = (1,3)):
        """
        Initializes a hypergraph representation of the BCCC-VulSCs-2023 dataset.
        
        :param train_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the training set.
        :type train_size: float
        :param val_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the validation set.
        :type val_size: float
        :param ngram_range: Tuple of two integers, representing the interval of
        n-gram sizes that will be used for generating hyperedges.
        :type ngram_range: tuple[int, int]
        """
        kaggle_path = kagglehub.dataset_download("bcccdatasets/bccc-vulscs-2023")
        secure_dir = os.path.join(kaggle_path, "Secure_SourceCodes/Secure_SourceCodes/")
        vuln_dir = os.path.join(kaggle_path, "Vulnerable_SourceCodes/Vulnerable_SourceCodes/")
        
        # load code and labels
        code = []
        labels = []
        for filename in next(os.walk(secure_dir))[2]:
            with open(os.path.join(secure_dir, filename), "r") as f:
                code.append(f.read())
                labels.append(0)
        for filename in next(os.walk(vuln_dir))[2]:
            with open(os.path.join(vuln_dir, filename), "r") as f:
                code.append(f.read())
                labels.append(1)
        code_labels = list(zip(code, labels))
        random.shuffle(code_labels)
        code, labels = zip(*code_labels)
        labels = torch.tensor(labels, dtype=bool)

        # generate edges
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=True, token_pattern='(?u)\\b\\w+\\b')
        X = vectorizer.fit_transform(code)
        rows, cols = X.nonzero()
        hyperedge_index = torch.vstack((torch.tensor(rows).long(), torch.tensor(cols).long()))
        del X, rows, cols
        
        # define hyperedge weights
        hyperedge_weight = torch.ones(((hyperedge_index[1].max().item())+1,))
        for k, v in vectorizer.vocabulary_.items():
            hyperedge_weight[v] = k.count(" ") + 1
        
        # merge duplicated hyperedges
        # First, sort hyperedge_index by hyperedge and then by hypernode, so that
        # all pairs from hyperedge_index with the same hyperedge are kept together,
        # sorted by hypernodes. Then, iterate through hyperedge_index,
        # storing all hypernodes that belong to the hyperedge, at the same time
        # comparing them to the hypernodes that belong to the previous hyperedge.
        hyperedge_index = hyperedge_index[:, torch.argsort(hyperedge_index[0])]
        hyperedge_index = hyperedge_index[:, torch.argsort(hyperedge_index[1])]
        previous_hyperedge = hyperedge_index[1, 0].item()
        current_hyperedge = hyperedge_index[1, 0].item()
        incident_hypernodes_prev = [] # list of hypernodes that belong to "previous_hyperedge"
        incident_hypernodes_cur = [] # list of hypernodes that belong to "current_hyperedge"
        hypernode_pointer = 0 # pointer for iterating through incident_hypernodes_prev
        similar = True # True if current hypernode is similar to previous hypernode, False otherwise
        for i in range(hyperedge_index.shape[1]):
            hn = hyperedge_index[0, i].item()
            he = hyperedge_index[1, i].item()
            if he!=current_hyperedge:
                if similar:
                    hyperedge_weight[current_hyperedge] += hyperedge_weight[previous_hyperedge]
                    hyperedge_weight[previous_hyperedge] = 0
                incident_hypernodes_prev = incident_hypernodes_cur
                incident_hypernodes_cur = []
                hypernode_pointer = 0
                previous_hyperedge = current_hyperedge
                current_hyperedge = he
                similar = True
            incident_hypernodes_cur.append(hn)
            if similar and len(incident_hypernodes_prev)>hypernode_pointer and incident_hypernodes_prev[hypernode_pointer]==hn:
                hypernode_pointer += 1
            else:
                similar = False
        # remove hyperedges with zero weight and reindex hyperedges
        hyperedge_index = hyperedge_index[:, hyperedge_weight[hyperedge_index[1]]!=0]
        new_hyperedge_indexes = torch.zeros((int(hyperedge_index[1].max().item()) + 1,), dtype=torch.long)
        new_hyperedge_indexes[hyperedge_weight!=0] = torch.arange((hyperedge_weight!=0).sum())
        hyperedge_index[1] = new_hyperedge_indexes[hyperedge_index[1]]
        hyperedge_weight = hyperedge_weight[hyperedge_weight!=0]
        
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        super().__init__(hyperedge_index, labels, hyperedge_weight, train_mask, val_mask, test_mask)