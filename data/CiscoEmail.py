from .hypergraph import Hypergraph
import torch

class CiscoEmailDataset(Hypergraph):
    """
    This is confidential dataset that belongs to Cisco. Hypernodes represent
    emails, and hyperedges represent shared attributes of emails, such as hyperlinks,
    attachments, etc. Label 1 stands for malicious email (phishing, spam, etc.), and 0 stands for benign email. 
    """
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2):
        """
        Initializes CiscoEmail dataset.

        :param train_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the training set.
        :type train_size: float
        :param val_size: Float number between 0 and 1, representing the portion
        of nodes that will be included in the validation set.
        :type val_size: float
        """
        raise Exception("CiscoEmail is a private dataset, currently unavailable publicly.")