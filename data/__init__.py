from .hypergraph import Hypergraph
from .CiscoEmail import CiscoEmailDataset
from .MAWI import MAWIDataset
from .MH_1M import MH1MDataset
from .SpamAssassin import SpamAssassinDataset

__all__ = ["Hypergraph", "CiscoEmailDataset", "MAWIDataset", "MH1MDataset", "SpamAssassinDataset", "DATASETS"]
DATASETS = {
    "CiscoEmail": CiscoEmailDataset,
    "MAWI": MAWIDataset,
    "MH-1M": MH1MDataset,
    "SpamAssassin": SpamAssassinDataset
}