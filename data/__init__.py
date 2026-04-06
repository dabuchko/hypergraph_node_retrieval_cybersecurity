"""
Module for loading datasets and constructing hypergraphs.
"""

from .hypergraph import Hypergraph
from .CiscoEmail import CiscoEmailDataset
from .MAWI import MAWIDataset
from .MH_1M import MH1MDataset
from .SpamAssassin import SpamAssassinDataset
from .BCCCVulSCs2023 import BCCCVulSCs2023Dataset

__all__ = ["Hypergraph", "CiscoEmailDataset", "MAWIDataset", "MH1MDataset", "SpamAssassinDataset", "BCCCVulSCs2023Dataset", "DATASETS"]

"""Dictionary mapping dataset names to their corresponding classes."""
DATASETS = {
    "CiscoEmail": CiscoEmailDataset,
    "MAWI": MAWIDataset,
    "MH-1M": MH1MDataset,
    "SpamAssassin": SpamAssassinDataset,
    "BCCC-VulSCs-2023": BCCCVulSCs2023Dataset
}