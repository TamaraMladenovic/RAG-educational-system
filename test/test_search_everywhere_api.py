# tests/test_search_everywhere_api.py
from pathlib import Path
from langchain_core.documents import Document

import pipeline.search_everywhere_api as sea


def test_search_everywhere_api_basic(monkeypatch):
    """
    Testiramo da agregator vrati ključevе za sve izvore
    i da poziva odgovarajuće load funkcije.
    """
    called = {
        "wikipedia": 0,
        "stackoverflow": 0,
        "openalex": 0,
        "pdf": 0,
    }

    def fake_load_wikipedia_by_query(query, lang, top_k, timeout):
        called["wikipedia"] += 1
        assert query == "rag systems"
        return [Document(page_content="wiki", metadata={"src": "wikipedia"})]

    def fake_load_stackoverflow_by_query(query, top_k, timeout, top_answers):
        called["stackoverflow"] += 1
        assert query == "rag systems"
        return [Document(page_content="so", metadata={"src": "stackoverflow"})]

    def fake_load_openalex_by_query(query, top_k, timeout):
        called["openalex"] += 1
        assert query == "rag systems"
        return [Document(page_content="openalex", metadata={"src": "openalex"})]

    def fake_search_local_pdfs_by_keywords(pdf_dir, query, page_level, top_k):
        called["pdf"] += 1
        assert query == "rag systems"
        assert isinstance(pdf_dir, (str, Path)) or pdf_dir is None
        return [Document(page_content="pdf", metadata={"src": "pdf"})]

    monkeypatch.setattr(sea, "load_wikipedia_by_query", fake_load_wikipedia_by_query)
    monkeypatch.setattr(sea, "load_stackoverflow_by_query", fake_load_stackoverflow_by_query)
    monkeypatch.setattr(sea, "load_openalex_by_query", fake_load_openalex_by_query)
    monkeypatch.setattr(sea, "search_local_pdfs_by_keywords", fake_search_local_pdfs_by_keywords)

    results = sea.search_everywhere_api("rag systems")

    # 1) ima sve ključevе
    assert set(results.keys()) == {"wikipedia", "stackoverflow", "openalex", "pdf"}

    # 2) svaki izvor je pozvan tačno jednom
    assert called == {
        "wikipedia": 1,
        "stackoverflow": 1,
        "openalex": 1,
        "pdf": 1,
    }

    # 3) svaki izvor ima po jedan rezultat
    assert len(results["wikipedia"]) == 1
    assert len(results["stackoverflow"]) == 1
    assert len(results["openalex"]) == 1
    assert len(results["pdf"]) == 1


def test_search_everywhere_respects_limits(monkeypatch):
    """
    Testiramo da se prosleđeni limits prelivaju u effective_limits
    (npr. openalex: 1, pdf: 2).
    """

    def fake_load_wikipedia_by_query(query, lang, top_k, timeout):
        # koristi default iz DEFAULT_LIMITS (3)
        assert top_k == sea.DEFAULT_LIMITS["wikipedia"]
        return []

    def fake_load_stackoverflow_by_query(query, top_k, timeout, top_answers):
        # koristi default iz DEFAULT_LIMITS (3)
        assert top_k == sea.DEFAULT_LIMITS["stackoverflow"]
        return []

    def fake_load_openalex_by_query(query, top_k, timeout):
        # ovde očekujemo override: openalex: 1
        assert top_k == 1
        return []

    def fake_search_local_pdfs_by_keywords(pdf_dir, query, page_level, top_k):
        # ovde očekujemo override: pdf: 2
        assert top_k == 2
        return []

    monkeypatch.setattr(sea, "load_wikipedia_by_query", fake_load_wikipedia_by_query)
    monkeypatch.setattr(sea, "load_stackoverflow_by_query", fake_load_stackoverflow_by_query)
    monkeypatch.setattr(sea, "load_openalex_by_query", fake_load_openalex_by_query)
    monkeypatch.setattr(sea, "search_local_pdfs_by_keywords", fake_search_local_pdfs_by_keywords)

    sea.search_everywhere_api(
        "anything",
        limits={"openalex": 1, "pdf": 2},
    )
