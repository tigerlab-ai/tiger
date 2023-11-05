import pandas as pd

"""
TigerRAG Dataloaders
============

This module serves as a repository of various data loaders

Example Usage:
    
    >>> from tigerrag.rag.base.loaders import TRAGDataLoader
    >>> trag_loader = TRAGDataLoader()
    >>> df = trag_loader.load_csv("path")
    
"""


class TigerRAGDataFrameLoader:
    def from_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
