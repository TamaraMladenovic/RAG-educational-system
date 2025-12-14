from __future__ import annotations

import re
import hashlib
from typing import List
from bs4 import BeautifulSoup

MIN_CHARS = 50

UA_JSON = {"User-Agent": "RAG-Search/0.2", "Accept": "application/json, */*"}
UA_HTML = {"User-Agent": "RAG-Search/0.2", "Accept": "text/html, */*"}

#Univerzalna funkcija za čišćenje teksta
def clean_text(s: str) -> str:
    s = s or ""
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    s = re.sub(r"[ \t\u00A0]+", " ", s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "form", "iframe", "aside"]
    ):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


def tokenize_query(q: str) -> List[str]:
    q = (q or "").lower()
    q = re.sub(r"[^a-z0-9A-ZčćžšđČĆŽŠĐ_\-\s]", " ", q)
    return [p for p in (w.strip() for w in q.split()) if p]
