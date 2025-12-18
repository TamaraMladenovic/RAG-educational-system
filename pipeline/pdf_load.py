from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from pipeline.rag_pipeline import RAGPipeline
from pipeline.pdf_search import load_pdfs


def main() -> None:

    load_dotenv(override=True)

    PDF_DIR = os.getenv("PDF_DIR")
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")

    if not PDF_DIR:
        raise RuntimeError("PDF_DIR nije definisan u .env")

    print(">>> PDF_DIR:", PDF_DIR)
    print(">>> FAISS_INDEX_DIR:", FAISS_INDEX_DIR)


    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF_DIR ne postoji: {pdf_dir}")

    rag = RAGPipeline(index_dir=FAISS_INDEX_DIR)


    pdf_docs = load_pdfs(pdf_dir, page_level=True)

    print(f">>> Loaded {len(pdf_docs)} PDF pages")

    if not pdf_docs:
        print(">>> Nema PDF dokumenata za ingest.")
        return

    # Ingest svake stranice
    for i, doc in enumerate(pdf_docs, start=1):
        source_path = doc.metadata.get("source", "")
        doc_id = os.path.basename(source_path) if source_path else "unknown_pdf"

        rag.ingest(
            text=doc.page_content,
            metadata={
                "source": "pdf",
                "doc_id": doc_id,
                "page": doc.metadata.get("page"),
            },
            save=False,   # čuvamo FAISS samo jednom
        )

        if i % 200 == 0:
            print(f">>> Ingested {i} pages")


    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    rag.store.save(FAISS_INDEX_DIR)

    print(">>> PDF INGEST ZAVRŠEN")
    print(f">>> FAISS index sačuvan u: {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
