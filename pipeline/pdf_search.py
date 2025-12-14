from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from pypdf import PdfReader

from pipeline.common import tokenize_query

MIN_CHARS = 50  

#Pretraživač PDF fajlova

def load_pdfs(root: str | Path, page_level: bool = True, dedup: bool = True) -> List[Document]:
    root = Path(root)
    docs: List[Document] = []
    seen = set()

    for pdf in root.glob("**/*.pdf"):
        try:
            reader = PdfReader(str(pdf))
        except Exception:
            continue

        if page_level:
            for i, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if len(text) < MIN_CHARS:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"type": "pdf", "source": str(pdf.resolve()), "page": i},
                    )
                )
        else:
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
            text = text.strip()
            if len(text) >= MIN_CHARS:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"type": "pdf", "source": str(pdf.resolve())},
                    )
                )

    return docs

def search_local_pdfs_by_keywords(
    pdf_dir: str | Path | None,
    query: str,
    *,
    page_level: bool = True,
    top_k: int = 20,
) -> List[Document]:
    
    if pdf_dir is None:
        pdf_dir = os.getenv("PDF_DIR")

    if not pdf_dir:
        return []

    pdf_dir = Path(pdf_dir)

    docs = load_pdfs(pdf_dir, page_level=page_level, dedup=True)
    if not docs:
        return []

    terms = tokenize_query(query)
    if not terms:
        return []

    scored: List[Tuple[float, Document]] = []
    for d in docs:
        text = (d.page_content or "").lower()
        score = sum(text.count(t) for t in terms)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]
