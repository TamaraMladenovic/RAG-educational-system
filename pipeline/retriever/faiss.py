from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import json

import faiss
import numpy as np

from pipeline.embeddings.base import EmbeddingModel


@dataclass
class IndexedDocument:    #Meta informacije o chunku koji je ubačen u FAISS

    doc_id: str           
    chunk_id: int         
    text: str             
    source: str           


class FaissStore:   

    #FAISS indeks i lista meta podataka

    def __init__(self,
                 embedding_model: EmbeddingModel,
                 index: Optional[faiss.Index] = None,
                 metadata: Optional[List[IndexedDocument]] = None) -> None:
        self.embedding_model = embedding_model
        dim = embedding_model.dimension

        # Ako index nije prosleđen, kreiramo novi L2 index
        self.index = index if index is not None else faiss.IndexFlatL2(dim)
        self.metadata: List[IndexedDocument] = metadata if metadata is not None else []


    # Dodavanje dokumenata - generiše embeddinge i dodaje u FAISS na osnovu IndexedDocument

    def add_chunks(self, chunks: List[IndexedDocument]) -> None:

        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_documents(texts)

        vec_np = np.array(vectors, dtype="float32")
        if vec_np.ndim == 1:
            vec_np = np.expand_dims(vec_np, axis=0)

        self.index.add(vec_np)
        self.metadata.extend(chunks)


    # Pretraga - vraća rezultate za prvih top_k chunkova
    def search(self, query: str, top_k: int = 5) -> List[Tuple[IndexedDocument, float]]:

        query_vec = self.embedding_model.embed_text(query)
        query_np = np.array([query_vec], dtype="float32")

        distances, indices = self.index.search(query_np, top_k)

        results: List[Tuple[IndexedDocument, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # FAISS vraća -1 ako nema dovoljno rezultata
            if idx >= len(self.metadata):
                continue
            results.append((self.metadata[idx], float(dist)))

        return results

    
    # Čuvanje / učitavanje - generiše novi folder data/index.faiss
    def save(self, dir_path: str) -> None:
        
        os.makedirs(dir_path, exist_ok=True)
        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "metadata.jsonl")

        faiss.write_index(self.index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps({
                    "doc_id": m.doc_id,
                    "chunk_id": m.chunk_id,
                    "text": m.text,
                    "source": m.source,
                }, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls,
             dir_path: str,
             embedding_model: EmbeddingModel) -> "FaissStore":     #Učitavanje iz postojećeg fajla

        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "metadata.jsonl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index.faiss found in {dir_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata.jsonl found in {dir_path}")

        index = faiss.read_index(index_path)

        metadata: List[IndexedDocument] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                metadata.append(IndexedDocument(
                    doc_id=obj["doc_id"],
                    chunk_id=int(obj["chunk_id"]),
                    text=obj["text"],
                    source=obj["source"],
                ))

        return cls(embedding_model=embedding_model, index=index, metadata=metadata)
