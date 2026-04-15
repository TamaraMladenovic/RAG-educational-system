from __future__ import annotations
from typing import List, Tuple
from pipeline.common import tokenize_query
from pipeline.retriever.faiss import IndexedDocument


def filter_by_keyword_overlap(
    query: str,
    results: List[Tuple[IndexedDocument, float]],
    min_overlap: int = 1,
) -> List[Tuple[IndexedDocument, float]]:
    """
    Lightweight post-retrieval reranker.
    Zadržava samo rezultate koji imaju minimalno preklapanje ključnih reči sa upitom.
    """

    query_tokens = set(tokenize_query(query))

    if not query_tokens:
        return results

    filtered = []

    for doc, dist in results:
        text_tokens = set(tokenize_query(doc.text))
        overlap = len(query_tokens & text_tokens)

        if overlap >= min_overlap:
            filtered.append((doc, dist))

    # Ako filter izbaci sve rezultate, vrati originalne (fallback sigurnost)
    return filtered if filtered else results