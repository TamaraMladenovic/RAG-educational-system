from __future__ import annotations

import os
from typing import List

import requests
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

#Google custom search klijent
def load_gcs_results(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 15,
) -> List[Document]:

    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        print("[GCS] Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in environment.")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": max(1, min(top_k, 10)),
    }

    try:
        r = requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        print(f"[GCS] Request error: {e}")
        return []

    if r.status_code != 200:
        print(f"[GCS] HTTP {r.status_code}: {r.text[:300]}")
        return []

    data = r.json()
    items = data.get("items", [])
    docs: List[Document] = []

    for it in items[:top_k]:
        title = it.get("title", "")
        snippet = it.get("snippet", "")
        link = it.get("link", "")
        text = f"{title}\n\n{snippet}"

        meta = {
            "source_type": "gcs",
            "title": title,
            "url": link,
        }
        docs.append(Document(page_content=text, metadata=meta))

    print(f"[GCS] Loaded {len(docs)} documents for query='{query}'")
    return docs
