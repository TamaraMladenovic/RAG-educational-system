from __future__ import annotations

import time
import urllib.parse
from typing import List, Sequence, Dict, Any, Tuple

import requests
from langchain_core.documents import Document

from pipeline.common import MIN_CHARS, clean_text, strip_html_to_text, hash_text

WIKI_HEADERS = {
    "User-Agent": "TamaraDiplomskiRAG/1.0 (https://github.com/TamaraMladenovic; contact: mladenovict58@gmail.com)"
}

#Klijent za pretragu wikipedije


def _generate_search_strategies(query: str) -> List[str]:

    q = (query or "").strip()
    strategies: List[str] = []

    if not q:
        return strategies

    # 1) original
    strategies.append(q)

    # 2) intitle: original
    strategies.append(f"intitle:{q}")

    # 3) navodnici
    strategies.append(f"\"{q}\"")

    # 4) plural -> singular heuristika (cows -> cow)
    if len(q.split()) == 1 and q.lower().endswith("s") and len(q) > 3:
        singular = q[:-1]
        strategies.append(singular)
        strategies.append(f"intitle:{singular}")
        strategies.append(f"\"{singular}\"")

    # ukloni duplikate 
    seen = set()
    out: List[str] = []
    for s in strategies:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def wiki_search_smart(
    query: str,
    *,
    lang: str = "en",
    top_k: int = 5,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], str]:

    api = f"https://{lang}.wikipedia.org/w/api.php"
    strategies = _generate_search_strategies(query)

    print(f">>> [WIKI_SMART] original query: {query}")
    print(f">>> [WIKI_SMART] strategies: {strategies}")

    for strategy in strategies:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": strategy,
            "srlimit": max(1, min(top_k, 50)),
            "format": "json",
            "utf8": "1",
        }
        r = requests.get(api, params=params, headers=WIKI_HEADERS, timeout=timeout)
        print(f">>> [WIKI_SMART] status={r.status_code} strategy='{strategy}' url={r.url}")

        if r.status_code != 200:
            # probaj sledeću strategiju
            continue

        hits = r.json().get("query", {}).get("search", [])
        if hits:
            print(f">>> [WIKI_SMART] FOUND {len(hits)} hits with strategy='{strategy}'")
            return [
                {
                    "title": h.get("title"),
                    "pageid": h.get("pageid"),
                    "snippet": strip_html_to_text(h.get("snippet", "")),
                    "timestamp": h.get("timestamp"),
                    "lang": lang,
                }
                for h in hits[:top_k]
            ], strategy

    # ako nijedna strategija nije uspela
    print(">>> [WIKI_SMART] No hits for any strategy.")
    return [], ""


# Fetch celog teksta
def wiki_fetch_plain(
    pageids: Sequence[int],
    *,
    lang: str = "en",
    timeout: int = 20,
) -> Dict[int, Dict[str, Any]]:
    if not pageids:
        return {}

    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "format": "json",
        "utf8": "1",
        "pageids": "|".join(str(p) for p in pageids),
    }
    r = requests.get(api, params=params, headers=WIKI_HEADERS, timeout=timeout)
    print(f">>> [WIKI_FETCH] status={r.status_code} url={r.url}")
    if r.status_code != 200:
        print(">>> [WIKI_FETCH] body:", r.text[:300])
        return {}

    pages = r.json().get("query", {}).get("pages", {})
    out: Dict[int, Dict[str, Any]] = {}
    for pid, obj in pages.items():
        try:
            pid_int = int(pid)
        except Exception:
            continue
        out[pid_int] = {
            "title": obj.get("title"),
            "extract": clean_text(obj.get("extract", "")),
            "fullurl": obj.get("canonicalurl") or obj.get("fullurl"),
        }
    return out


# Glavna funkcija koju koristi RAG
def load_wikipedia_by_query(
    query: str,
    *,
    lang: str = "en",
    top_k: int = 5,
    timeout: int = 20,
    dedup: bool = True,
) -> List[Document]:

    hits, used_strategy = wiki_search_smart(query, lang=lang, top_k=top_k, timeout=timeout)
    print(f">>> [WIKI_LOAD] used_strategy='{used_strategy}', raw_hits={len(hits)}")

    pageids = [h["pageid"] for h in hits if h.get("pageid")]
    details = wiki_fetch_plain(pageids, lang=lang, timeout=timeout)

    docs: List[Document] = []
    seen = set()

    for h in hits:
        pid = h.get("pageid")
        det = details.get(pid, {})
        text = det.get("extract", "")
        if len(text) < MIN_CHARS:
            print(f">>> [WIKI_LOAD] Skipping short article: title='{h.get('title')}' len={len(text)}")
            continue

        if dedup:
            hh = hash_text(text)
            if hh in seen:
                print(f">>> [WIKI_LOAD] Duplicate article skipped: title='{h.get('title')}'")
                continue
            seen.add(hh)

        meta = {
            "type": "wikipedia",
            "title": det.get("title") or h.get("title"),
            "url": det.get("fullurl")
            or (
                f"https://{lang}.wikipedia.org/wiki/"
                + urllib.parse.quote((h.get("title") or "").replace(" ", "_"))
            ),
            "pageid": pid,
            "lang": lang,
            "snippet": h.get("snippet"),
            "fetched_at": int(time.time()),
        }
        docs.append(Document(page_content=text, metadata=meta))

    print(f">>> [WIKI_LOAD] Final docs count: {len(docs)}")
    return docs
