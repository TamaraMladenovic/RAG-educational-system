from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import requests
from langchain_core.documents import Document

from pipeline.common import MIN_CHARS, UA_JSON, clean_text, hash_text

from dotenv import load_dotenv

load_dotenv()

#Povlači podatke sa OpenAlex sajta 
def _uninvert_openalex(inv: Dict[str, List[int]] | None) -> str:
    #Pretvara inverted_index format u normalan tekst

    if not inv:
        return ""
    max_pos = max((max(v) for v in inv.values() if v), default=0)
    words = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for pos in positions:
            words[pos] = token
    return clean_text(" ".join(w for w in words if w))


def openalex_search(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 20,
) -> List[Dict[str, Any]]:    #Vraća listu radova
    
    api = "https://api.openalex.org/works"
    mailto = os.getenv("OPENALEX_MAILTO")

    params = {"search": query, "per_page": max(1, min(top_k, 25))}
    if mailto:
        params["mailto"] = mailto

    r = requests.get(api, params=params, headers=UA_JSON, timeout=timeout)
    if r.status_code != 200:
        return []

    results = []
    for it in r.json().get("results", [])[:top_k]:
        results.append(
            {
                "title": it.get("title"),
                "abstract": _uninvert_openalex(it.get("abstract_inverted_index")),
                "url": (
                    it.get("primary_location", {})
                    .get("source", {})
                    .get("host_page_url")
                    or it.get("primary_location", {}).get("landing_page_url")
                ),
                "doi": it.get("doi"),
                "year": (it.get("from_publication_date") or "")[:4]
                or it.get("publication_year"),
                "source": "openalex",
            }
        )
    return results


def load_openalex_by_query(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 20,
) -> List[Document]:    #Konvertuje OpenAlex rezultate u dokumente

    recs = openalex_search(query, top_k=top_k, timeout=timeout)

    docs: List[Document] = []
    seen = set()

    for r in recs:
        content = clean_text(f"# {r.get('title','')}\n\n{r.get('abstract','')}".strip())
        if len(content) < MIN_CHARS:
            continue

        hh = hash_text(content)
        if hh in seen:
            continue
        seen.add(hh)

        meta = {
            "type": "openalex",
            "title": r.get("title"),
            "url": r.get("url"),
            "doi": r.get("doi"),
            "year": r.get("year"),
            "fetched_at": int(time.time()),
        }

        docs.append(Document(page_content=content, metadata=meta))

    return docs
