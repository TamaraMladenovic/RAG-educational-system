from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import os

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .search_everywhere import search_everywhere
from .chunking import chunk_documents
from pipeline.embeddings.local import LocalHFEmbeddingModel
from .retriever.faiss import FaissStore, IndexedDocument
from .context_formatter import build_prompt
from pipeline.llm.factory import get_llm_adapter
from .query_rewriter import rewrite_query_for_search
from .keyword_overlap import filter_by_keyword_overlap


class RAGPipeline:
    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.llm = get_llm_adapter()
        self.embedding_model = LocalHFEmbeddingModel()

        self.index_dir = index_dir or os.getenv("FAISS_INDEX_DIR", "data/faiss_index")

        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.jsonl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print(f">>> [FAISS] Loading existing index from: {self.index_dir}")
            self.store = FaissStore.load(self.index_dir, embedding_model=self.embedding_model)
        else:
            print(f">>> [FAISS] Creating NEW empty index in: {self.index_dir}")
            self.store = FaissStore(embedding_model=self.embedding_model)

    # ===============================
    # INGEST (offline PDF indexing)
    # ===============================
    def ingest(
        self,
        text: str,
        metadata: Dict | None = None,
        save: bool = True,
    ) -> None:
        meta = metadata or {}
        base_doc = Document(page_content=text, metadata=meta)

        # ✅ drop_noise=True da filtrira TOC/literaturu
        chunked_docs: List[Document] = chunk_documents([base_doc])
        if not chunked_docs:
            return

        indexed_chunks: List[IndexedDocument] = []
        for i, ch in enumerate(chunked_docs):
            ch_meta = ch.metadata or {}

            doc_id = (
                ch_meta.get("doc_id")
                or meta.get("doc_id")
                or "unknown_doc"
            )

            source = (
                ch_meta.get("source")
                or ch_meta.get("source_type")
                or meta.get("source")
                or "generic"
            )

            indexed_chunks.append(
                IndexedDocument(
                    doc_id=str(doc_id),
                    chunk_id=i,
                    text=ch.page_content,
                    source=str(source),
                )
            )

        self.store.add_chunks(indexed_chunks)

        if save:
            os.makedirs(self.index_dir, exist_ok=True)
            self.store.save(self.index_dir)

    # ===============================
    # LIVE SEARCH (Google + Wiki + SO + OpenAlex)
    # ===============================
    def search_live_sources(self, query: str, limit: int = 5):

        search_query = rewrite_query_for_search(self.llm, query)

        results = search_everywhere(
            query=search_query,
            lang="en",
            limits={
                "gcs": limit,
                "wikipedia": limit,
                "stackoverflow": min(3, limit),
                "openalex": min(3, limit),
            },
            timeout=20,
        )

        return results

    # ===============================
    # FAISS
    # ===============================
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Tuple[IndexedDocument, float]]:
        return self.store.search(query, top_k=top_k)

    # ===============================
    # GENERATE
    # ===============================
    def generate(self, query: str, chunk_dicts: List[Dict[str, Any]]) -> str:
        prompt = build_prompt(query, chunk_dicts)

        llm: Any = self.llm
        if hasattr(llm, "generate") and callable(getattr(llm, "generate")):
            return llm.generate(prompt)
        if hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
            return llm.invoke(prompt)
        if callable(llm):
            return llm(prompt)

        raise TypeError(
            f"LLM adapter tipa {type(llm).__name__} "
            f"nema .generate(), .invoke(), ni __call__."
        )

    # ===============================
    # GLAVNI RAG PIPELINE
    # ===============================
    def run(self, query: str, top_k: int = 3) -> Dict:

        # ---- LIVE ----
        live_results = self.search_live_sources(query)

        live_docs: List[Document] = []
        for source, docs in live_results.items():
            for doc in docs:
                meta = dict(doc.metadata or {})
                meta.setdefault("source_type", source)
                meta.setdefault("source", source)
                doc.metadata = meta
                live_docs.append(doc)

                # chunk live
        live_chunks: List[Document] = []
        if live_docs:
            live_chunks = chunk_documents(live_docs)

        preferred_live_sources = {"gcs"}
        filtered_live_chunks: List[Document] = []
        for ch in live_chunks:
            st = (ch.metadata or {}).get("source_type")
            if st in preferred_live_sources:
                filtered_live_chunks.append(ch)
        filtered_live_chunks = filtered_live_chunks[:2]

        # ---- FAISS ----
        faiss_results = self.retrieve_context(query, top_k=top_k)

        faiss_results = filter_by_keyword_overlap(
            query,
            faiss_results,
            min_overlap=2, 
        )

        # ---- BUILD CHUNK DICTS (za prompt) ----
        chunk_dicts: List[Dict[str, Any]] = []

        # LIVE -> chunk_dicts sa title/url
        for ch in filtered_live_chunks:
            meta = ch.metadata or {}
            chunk_dicts.append(
                {
                    "text": ch.page_content,
                    "source": meta.get("source_type", "live"),
                    "metadata": {
                        "title": meta.get("title"),
                        "url": meta.get("url"),
                        "source_type": meta.get("source_type"),
                    },
                }
            )

        # FAISS -> chunk_dicts sa doc_id/chunk_id/dist
        for doc, dist in faiss_results[:top_k]:
            chunk_dicts.append(
                {
                    "text": doc.text,
                    "source": f"faiss:{doc.doc_id}",
                    "metadata": {
                        "doc_id": doc.doc_id,
                        "chunk_id": doc.chunk_id,
                        "distance": dist,
                    },
                }
            )

        # ---- GENERATE ----
        answer = self.generate(query, chunk_dicts=chunk_dicts)

        # ---- RETRIEVED DOCS (za evaluaciju) ----
        live_retrieved_docs: List[Dict[str, Any]] = []
        for source, docs in live_results.items():
            for d in docs[:top_k]:
                meta = dict(d.metadata or {})
                live_retrieved_docs.append(
                    {
                        "source_type": source,
                        "source": meta.get("url") or meta.get("source"),
                        "title": meta.get("title"),
                        "text": d.page_content,
                        "score": meta.get("score"),
                    }
                )

        return {
            "query": query,
            "live_results": live_results,
            "retrieved_chunks": [
                {
                    "doc_id": doc.doc_id,
                    "chunk_id": doc.chunk_id,
                    "source": doc.source,
                    "text": doc.text,
                    "distance": dist,
                }
                for (doc, dist) in faiss_results
            ],

            "final_answer": answer,
            "answer": answer,

            "retrieved_docs": live_retrieved_docs + [
                {
                    "source": doc.source,
                    "source_type": "faiss",
                    "doc_id": doc.doc_id,
                    "chunk_id": doc.chunk_id,
                    "text": doc.text,
                    "score": dist,
                }
                for (doc, dist) in faiss_results
            ],
        }