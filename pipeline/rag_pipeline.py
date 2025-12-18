from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import os

#Ukoliko ima nepoklapanja u LLM verzijma
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

#Ceo RAG spojen
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


    def ingest(
        self,
        text: str,
        metadata: Dict | None = None,
        save: bool = True,
    ) -> None:
        meta = metadata or {}
        base_doc = Document(page_content=text, metadata=meta)

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
                ch_meta.get("source_type")
                or ch_meta.get("source")
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


    # API pretraga
    def search_live_sources(self, query: str, limit: int = 5):

        search_query = rewrite_query_for_search(self.llm, query)

        print(f">>> [LIVE SEARCH] original question: {query}")
        print(f">>> [LIVE SEARCH] rewritten search query in english: {search_query}")

        results = search_everywhere(
            query=search_query,
            lang="en",
            limits={"wikipedia": limit},
            timeout=20,
        )

        wiki_docs = results.get("wikipedia", [])
        print(f">>> [LIVE SEARCH] wikipedia docs count: {len(wiki_docs)}")

        return results


    # FAISS 
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Tuple[IndexedDocument, float]]:
        return self.store.search(query, top_k=top_k)


    # LLM generisanje
    def generate(self, query: str, context_blocks: List[str]) -> str:
        print(">>> [DEBUG] generate() pozvan, context_blocks:", len(context_blocks))

        chunk_dicts = [
            {
                "text": text,
                "source": "mixed",
                "metadata": {},
            }
            for text in context_blocks
        ]

        prompt = build_prompt(query, chunk_dicts)

        llm: Any = self.llm
        if hasattr(llm, "generate") and callable(getattr(llm, "generate")):
            return llm.generate(prompt)
        if hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
            return llm.invoke(prompt)
        if callable(llm):
            return llm(prompt)

        raise TypeError(
            f"LLM adapter objekat tipa {type(llm).__name__} "
            f"nema ni .generate(), ni .invoke(), ni __call__."
        )


    # Glavni RAG pipeline
    # ----------------------------------------------------------------------
    def run(self, query: str, top_k: int = 5) -> Dict:

        live_results = self.search_live_sources(query)

        live_docs: List[Document] = []
        for source, docs in live_results.items():
            for doc in docs:
                meta = dict(doc.metadata or {})
                meta.setdefault("source_type", source)
                doc.metadata = meta
                live_docs.append(doc)

        live_context: List[str] = []
        if live_docs:
            live_chunks = chunk_documents(live_docs)
            live_context = [ch.page_content for ch in live_chunks]

        faiss_results = self.retrieve_context(query, top_k=top_k)
        faiss_context = [doc.text for (doc, dist) in faiss_results]

        final_context = live_context[:2] + faiss_context[:2]

        ####### DEBUG ########
        print("\n" + "=" * 60)
        print(">>> [DEBUG] FINAL_CONTEXT length:", len(final_context))
        for i, ctx in enumerate(final_context):
            print(f"\n--- CONTEXT {i+1} ---\n")
            print(ctx[:500])
        print("=" * 60 + "\n")

        answer = self.generate(query, context_blocks=final_context)

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
        }
