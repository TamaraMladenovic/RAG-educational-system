from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    
    #Apstraktna klasa za lakše menjanje između cloud i lokalnih embeddings-a 
    #Koristi se samo lokalni, ali će na ovaj način biti lakše dodati i cloud ukoliko zatreba

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:  #String u vektor
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:  #Embedding liste chunkova
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:   #Dimenzija vektora
        raise NotImplementedError
