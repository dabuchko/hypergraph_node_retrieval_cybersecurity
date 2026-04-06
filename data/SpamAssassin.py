from .hypergraph import Hypergraph
from datasets import load_dataset
from tokenizers import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import torch

class SpamAssassinDataset(Hypergraph):
    """
    Dataset class for constrcuting a hypergraph representation of the original SpamAssassin dataset.
    Each hypernode represents email, and each hyperedge represents an n-gram,
    where n-gram range is provided in initialization argument. Hypernode label is 1 if
    the email is malicious (spam, phishing, etc.) and 0 otherwise.
    Hyperedge weight is defined as the number of tokens in the n-gram.
    """
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2, ngram_range: tuple[int, int] = (1,3)):
        """
        Initializes a hypergraph representation of the SpamAssassin dataset.
        
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
        ds = load_dataset("talby/spamassassin", "text")
        tokenizer = Tokenizer.from_pretrained("openai-community/gpt2")
        labels = torch.tensor(ds["train"]["label"], dtype=bool)
        # generate edges
        vectorizer = CountVectorizer(ngram_range=ngram_range, tokenizer=lambda x: tokenizer.encode(x).tokens,
                                     token_pattern=None, binary=True)
        X = vectorizer.fit_transform(ds["train"]["text"])
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