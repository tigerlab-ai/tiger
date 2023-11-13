import pandas as pd

"""
TigerDA Dataloaders
============

This module serves as a repository of various data loaders

Example Usage:
    
    >>> from tigerda.rag.base.loaders import TigerDADataFrameLoader
    >>> tda_loader = TigerDADataFrameLoader()
    >>> df = trag_loader.load_csv("path")
    
"""


class TigerDADataFrameLoader:
    def from_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
