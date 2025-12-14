from __future__ import annotations
from typing import Dict, List
from pathlib import Path

from langchain_core.documents import Document
from pipeline.search_everywhere_api import search_everywhere_api

def search_everywhere(
    query: str,
    *,
    lang: str = "en",
    pdf_dir: str | Path | None = None,
    limits: Dict[str, int] | None = None,
    timeout: int = 20,
    page_level_pdf: bool = True,
) -> Dict[str, List[Document]]:
    return search_everywhere_api(
        query=query,
        lang=lang,
        pdf_dir=pdf_dir,
        limits=limits,
        timeout=timeout,
        page_level_pdf=page_level_pdf,
    )
