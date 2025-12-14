from __future__ import annotations
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from pipeline.embeddings.base import EmbeddingModel


class LocalHFEmbeddingModel():

    #Lokalni embedding model - sentence-transformers, all-MiniLM-L6-v2

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: Optional[str] = None) -> None:
        
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

        # Jedan probni encode da se dobije dimenzija
        test_vec = self._model.encode("test", convert_to_numpy=True)
        self._dimension = int(test_vec.shape[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        mat = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [row.tolist() for row in mat]
