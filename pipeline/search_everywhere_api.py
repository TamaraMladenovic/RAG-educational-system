from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Callable

from dotenv import load_dotenv
from langchain_core.documents import Document

from pipeline.wikipedia_client import load_wikipedia_by_query
from pipeline.stackoverflow_client import load_stackoverflow_by_query
from pipeline.openalex_client import load_openalex_by_query
from pipeline.pdf_search import load_pdfs, search_local_pdfs_by_keywords
from pipeline.google_client import load_gcs_results

load_dotenv(override=True)


DEFAULT_LIMITS: Dict[str, int] = {
    "gcs": 5, 
    "stackoverflow": 3,
    "openalex": 5,
    "pdf": 20,
    "wikipedia": 3,    
}

#Glavni modul za pretragu, koristi query i pretražuje uz pomoć svih klijenata
def _safe_call(loader: Callable[[], List[Document]]) -> List[Document]:
    try:
        return loader()
    except Exception:
        return []


def search_everywhere_api(
    query: str,
    *,
    lang: str = "en",
    pdf_dir: str | Path | None = None,
    limits: Dict[str, int] | None = None,
    timeout: int = 20,
    page_level_pdf: bool = True,
) -> Dict[str, List[Document]]:
    effective_limits = {**DEFAULT_LIMITS, **(limits or {})}

    loaders: Dict[str, Callable[[], List[Document]]] = {
        "gcs": lambda: load_gcs_results(        
            query,
            top_k=effective_limits["gcs"],
            timeout=timeout,
        ),
        "stackoverflow": lambda: load_stackoverflow_by_query(
            query,
            top_k=effective_limits["stackoverflow"],
            timeout=timeout,
            top_answers=2,
        ),
        "openalex": lambda: load_openalex_by_query(
            query,
            top_k=effective_limits["openalex"],
            timeout=timeout,
        ),
        "wikipedia": lambda: load_wikipedia_by_query(
            query,
            lang=lang,
            top_k=effective_limits["wikipedia"],
            timeout=timeout,
        ),
        "pdf": lambda: search_local_pdfs_by_keywords(
            pdf_dir,
            query,
            page_level=page_level_pdf,
            top_k=effective_limits["pdf"],
        ),
    }

    return {name: _safe_call(fn) for name, fn in loaders.items()}
