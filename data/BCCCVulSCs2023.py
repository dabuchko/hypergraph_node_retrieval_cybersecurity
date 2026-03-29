from .hypergraph import Hypergraph
import kagglehub
import torch
import os
from sklearn.feature_extraction.text import CountVectorizer
import random

class BCCCVulSCs2023Dataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2, ngram_range=(1,3)):
        kaggle_path = kagglehub.dataset_download("bcccdatasets/bccc-vulscs-2023")
        secure_dir = os.path.join(kaggle_path, "Secure_SourceCodes/Secure_SourceCodes/")
        vuln_dir = os.path.join(kaggle_path, "Vulnerable_SourceCodes/Vulnerable_SourceCodes/")
        
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
        # generate mask
        mask = torch.rand((labels.shape[0],))
        train_mask = mask < train_size
        val_mask = torch.logical_and(train_size < mask, mask < (train_size+val_size))
        test_mask = torch.logical_and(~train_mask, ~val_mask)
        # define hyperedge weights
        hyperedge_weight = torch.ones((X.shape[1],))
        for k, v in vectorizer.vocabulary_.items():
            hyperedge_weight[v] = k.count(" ") + 1
        super().__init__(hyperedge_index, labels, hyperedge_weight, train_mask, val_mask, test_mask)