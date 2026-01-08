from .hypergraph import Hypergraph
from datasets import load_dataset
from tokenizers import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import torch

class SpamAssassinDataset(Hypergraph):
    def __init__(self, train_size=0.6, val_size=0.2, ngram_range=(1,3)):
        ds = load_dataset("talby/spamassassin", "text")
        tokenizer = Tokenizer.from_pretrained("openai-community/gpt2")
        labels = torch.tensor(ds["train"]["label"], dtype=bool)
        # generate edges
        vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=lambda x: tokenizer.encode(x).tokens,
                                     token_pattern=None, binary=True)
        X = vectorizer.fit_transform(ds["train"]["text"])
        rows, cols = X.nonzero()
        hyperedge_index = torch.vstack((torch.tensor(rows), torch.tensor(cols)))
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