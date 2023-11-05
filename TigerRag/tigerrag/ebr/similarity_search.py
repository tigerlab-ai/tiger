import faiss
import numpy.typing as npt
from numpy.typing import NDArray


class FaissFlatL2Search:
    def __init__(self, embedding_dim: int) -> None:
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_to_index(self, embeddings: NDArray) -> None:
        self.index.add(embeddings)

    def search(self, query: NDArray, k: int) -> tuple[npt.NDArray, npt.NDArray]:
        distances, labels = self.index.search(query.reshape(1, -1), k=k)  # Retrieve top 5 similar movies

        return distances, labels
