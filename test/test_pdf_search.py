# tests/test_pdf_search.py
from pathlib import Path
from langchain_core.documents import Document

import pipeline.pdf_search as pdf_search


def test_search_uses_env_pdf_dir_when_none(monkeypatch, tmp_path):
    """
    Ako pdf_dir nije prosleđen, funkcija treba da koristi PDF_DIR iz okruženja.
    Proveravamo da se load_pdfs pozove sa tom putanjom.
    """
    fake_dir = tmp_path / "pdfs"
    fake_dir.mkdir()

    called = {}

    def fake_getenv(name, default=None):
        if name == "PDF_DIR":
            return str(fake_dir)
        return default

    def fake_load_pdfs(root, page_level=True, dedup=True):
        # zabeleži sa čim je pozvano
        called["root"] = Path(root)
        return []

    monkeypatch.setattr(pdf_search.os, "getenv", fake_getenv)
    monkeypatch.setattr(pdf_search, "load_pdfs", fake_load_pdfs)

    res = pdf_search.search_local_pdfs_by_keywords(
        pdf_dir=None,
        query="test",
        page_level=True,
        top_k=10,
    )

    assert res == []
    assert "root" in called
    assert called["root"] == fake_dir


def test_search_ranks_by_keyword_frequency(monkeypatch):
    """
    Proveravamo da li se dokument sa više pojavljivanja termina rangira više.
    """
    docs = [
        Document(page_content="ovo je test test dokument", metadata={"id": 1}),
        Document(page_content="ovo je test dokument", metadata={"id": 2}),
    ]

    def fake_load_pdfs(root, page_level=True, dedup=True):
        return docs

    # ne zanima nas realni env ovde
    def fake_getenv(name, default=None):
        return "/fake/path"

    monkeypatch.setattr(pdf_search, "load_pdfs", fake_load_pdfs)
    monkeypatch.setattr(pdf_search.os, "getenv", fake_getenv)

    results = pdf_search.search_local_pdfs_by_keywords(
        pdf_dir=None,
        query="test",
        page_level=True,
        top_k=10,
    )

    # očekujemo da dokument sa 2x "test" bude prvi
    assert len(results) == 2
    assert results[0].metadata["id"] == 1
    assert results[1].metadata["id"] == 2
