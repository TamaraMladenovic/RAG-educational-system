from __future__ import annotations

import os
import time
from typing import List, Sequence, Dict, Any

import requests
from langchain_core.documents import Document

from pipeline.common import MIN_CHARS, UA_JSON, strip_html_to_text, clean_text, hash_text

from dotenv import load_dotenv

load_dotenv()

#Klijent za pretragu StackOverflow
def so_search(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 20,
) -> List[int]:
    api = "https://api.stackexchange.com/2.3/search/advanced"
    key = os.getenv("STACKEXCHANGE_KEY")

    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": max(1, min(top_k, 50)),
    }
    if key:
        params["key"] = key

    r = requests.get(api, params=params, headers=UA_JSON, timeout=timeout)
    if r.status_code != 200:
        return []

    items = r.json().get("items", [])
    return [it.get("question_id") for it in items[:top_k] if it.get("question_id")]


def so_fetch_qna(
    question_ids: Sequence[int],
    *,
    top_answers: int = 2,
    timeout: int = 20,
) -> List[Document]:
    if not question_ids:
        return []

    key = os.getenv("STACKEXCHANGE_KEY")

    #Pitanja
    base = "https://api.stackexchange.com/2.3/questions/"
    q_api = base + ";".join(str(i) for i in question_ids)
    q_params = {
        "order": "desc",
        "sort": "activity",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": 50,
    }
    if key:
        q_params["key"] = key
    q_resp = requests.get(q_api, params=q_params, headers=UA_JSON, timeout=timeout)
    if q_resp.status_code != 200:
        return []

    q_map = {it["question_id"]: it for it in q_resp.json().get("items", [])}

    #Odgovori
    a_api = base + ";".join(str(i) for i in question_ids) + "/answers"
    a_params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": top_answers,
    }
    if key:
        a_params["key"] = key
    a_resp = requests.get(a_api, params=a_params, headers=UA_JSON, timeout=timeout)
    a_items = a_resp.json().get("items", []) if a_resp.status_code == 200 else []

    ans_by_q: Dict[int, List[str]] = {}
    for a in a_items:
        qid = a.get("question_id")
        body_html = a.get("body") or ""
        body_txt = strip_html_to_text(body_html)
        ans_by_q.setdefault(qid, []).append(body_txt)

    docs: List[Document] = []
    seen = set()

    for qid in question_ids:
        q = q_map.get(qid)
        if not q:
            continue

        title = q.get("title") or f"SO Question {qid}"
        q_body = strip_html_to_text(q.get("body") or "")

        combined = f"# {title}\n\n## Question\n{q_body}"
        answers = ans_by_q.get(qid, [])
        if answers:
            combined += "\n\n## Top Answers\n" + "\n\n---\n\n".join(
                answers[:top_answers]
            )

        combined = clean_text(combined)

        if len(combined) < MIN_CHARS:
            continue

        hh = hash_text(combined)
        if hh in seen:
            continue
        seen.add(hh)

        url = q.get("link") or f"https://stackoverflow.com/questions/{qid}"
        meta = {
            "type": "stack_overflow",
            "title": title,
            "url": url,
            "site": "stack_overflow",
            "fetched_at": int(time.time()),
        }
        docs.append(Document(page_content=combined, metadata=meta))

    return docs


def load_stackoverflow_by_query(
    query: str,
    *,
    top_k: int = 5,
    timeout: int = 20,
    top_answers: int = 2,
) -> List[Document]:
    ids = so_search(query, top_k=top_k, timeout=timeout)
    return so_fetch_qna(ids, top_answers=top_answers, timeout=timeout)
