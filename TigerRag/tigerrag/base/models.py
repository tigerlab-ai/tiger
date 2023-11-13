from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from transformers import (BertModel, BertTokenizer, RobertaModel,
                          RobertaTokenizer, XLNetModel, XLNetTokenizer)

"""
TigerRAG Models
============

This module serves as a repository for supported Models 
Example Usage:
    >>> from tigerrag.rag.base.models import TigerRAGEmbeddingModel
    >>> trag_bert_model = TigerRAGEmbeddingModel(EmbeddingModel.BERT)
    >>> query_embedding = trag_bert_model.get_embedding_from_text("query_text")
    >>> df_series_embeddings = trag_bert_model.get_embedding_from_series("df_series")
    
"""


class EmbeddingModel(Enum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3


class TigerRAGEmbeddingModel:
    def __init__(self, model_id: EmbeddingModel):
        if model_id is EmbeddingModel.BERT:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif model_id is EmbeddingModel.ROBERTA:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif model_id == EmbeddingModel.XLNET:
            self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
            self.model = XLNetModel.from_pretrained("xlnet-base-cased")
        else:
            raise ValueError("Unsupported Model")

    def get_embedding_from_text(self, text: str) -> npt.NDArray:
        """Returns the BERT embedding for a given text."""
        tokens = self.tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=100, truncation=True)
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state
        return embeddings.mean(1).numpy()

    def get_embedding_from_series(self, pd_series: pd.Series) -> npt.NDArray:
        embeddings = np.vstack(pd_series.apply(self.get_embedding_from_text))
        return embeddings
