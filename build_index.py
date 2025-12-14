# build_index.py

from pathlib import Path
from typing import List, Dict

from pipeline.chunking import chunk_text  # prilagodi imenu svoje funkcije
from pipeline.embeddings.local import LocalHFEmbeddingModel
from pipeline.retriever.faiss import FaissStore, IndexedDocument

# ✅ OVO prilagodi svom pdf_search modulu:
from pipeline import pdf_search  # pretpostavka: imaš ovaj modul


DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "indices" / "pdf_index"


def load_pdf_texts() -> List[Dict]:
    """
    Vrati listu dictova: {doc_id, text, source}
    Ovde odlučuješ koje PDF-ove indeksiraš.
    """
    pdf_folder = Path("data/pdfs")  # npr. ovde držiš PDF-ove
    docs: List[Dict] = []

    for pdf_path in pdf_folder.glob("*.pdf"):
        # ❗ OVO prilagodi: zavisi kako si napravila pdf_search
        # Npr. ako imaš funkciju pdf_search.load_pdf_text(path: str) -> str
        text = pdf_search.load_pdf_text(str(pdf_path))

        docs.append({
            "doc_id": pdf_path.name,   # može i apsolutna putanja ili neki ID
            "text": text,
            "source": "pdf",
        })

    return docs


def build_index() -> None:
    print("🔹 Učitavam PDF dokumente...")
    raw_docs = load_pdf_texts()

    embedding_model = LocalHFEmbeddingModel()
    store = FaissStore(embedding_model=embedding_model)

    all_chunks: List[IndexedDocument] = []

    print("🔹 Chunkujem dokumente i pripremam za indeksiranje...")
    for doc in raw_docs:
        doc_id = doc["doc_id"]
        source = doc["source"]
        full_text = doc["text"]

        # 1) CHUNKING
        chunks_str: List[str] = chunk_text(full_text)

        # 2) Pakovanje u IndexedDocument
        for i, ch in enumerate(chunks_str):
            all_chunks.append(IndexedDocument(
                doc_id=doc_id,
                chunk_id=i,
                text=ch,
                source=source,
            ))

    print(f"🔹 Ukupno chunkova za indeksiranje: {len(all_chunks)}")

    # 3) Embedding + FAISS
    store.add_chunks(all_chunks)

    print(f"🔹 Čuvam indeks u: {INDEX_DIR}")
    store.save(str(INDEX_DIR))

    print("✅ Gotovo! FAISS indeks je napravljen.")


if __name__ == "__main__":
    build_index()

