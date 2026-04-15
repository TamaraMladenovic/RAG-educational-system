from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokens(s: str) -> set[str]:
    # radi i za sr/en (čćđšž ostaju)
    s = _norm(s)
    toks = re.findall(r"[a-z0-9čćđšž]+", s, flags=re.IGNORECASE)
    return {t for t in toks if len(t) >= 3}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def evidence_to_list(evidence: Any) -> List[str]:
    """
    evidence u gold_datasetu može biti:
    - string
    - lista stringova
    - lista dict-ova (npr {"text": "..."} ili {"content": "..."})
    """
    if evidence is None:
        return []

    if isinstance(evidence, str):
        ev = evidence.strip()
        return [ev] if ev else []

    if isinstance(evidence, list):
        out: List[str] = []
        for item in evidence:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "snippet", "evidence"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
        return out

    return []


def doc_text(doc: Any) -> str:
    """
    retrieved_docs kod tebe su dict sa 'text' najčešće.
    """
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "page_content", "content", "chunk", "snippet"):
            v = doc.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def doc_matches_any_evidence(doc: Any, evidence_list: List[str], threshold: float) -> bool:
    dt = doc_text(doc)
    if not dt or not evidence_list:
        return False
    return any(jaccard(dt, ev) >= threshold for ev in evidence_list)


@dataclass
class PerQuery:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    first_relevant_rank: int | None
    relevant_in_top_k: int
    k: int


def compute_metrics_for_item(
    retrieved_docs: List[Dict[str, Any]],
    evidence: Any,
    k: int,
    threshold: float = 0.12,
) -> PerQuery:
    ev_list = evidence_to_list(evidence)
    topk = (retrieved_docs or [])[:k]

    rel_flags: List[bool] = [
        doc_matches_any_evidence(d, ev_list, threshold=threshold) for d in topk
    ]
    rel_count = sum(1 for x in rel_flags if x)

    precision = (rel_count / k) if k > 0 else 0.0

    # Minimalna i stabilna verzija Recall@k:
    # ako evidence postoji -> recall = 1 ako ima bar jedan relevantan doc u top-k, inače 0
    if len(ev_list) == 0:
        recall = 0.0
    else:
        recall = 1.0 if rel_count > 0 else 0.0

    first_rank = None
    for idx, ok in enumerate(rel_flags, start=1):
        if ok:
            first_rank = idx
            break
    mrr = (1.0 / first_rank) if first_rank else 0.0

    return PerQuery(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        first_relevant_rank=first_rank,
        relevant_in_top_k=rel_count,
        k=k,
    )


def aggregate(per_query: List[PerQuery]) -> Dict[str, float]:
    if not per_query:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    n = len(per_query)
    return {
        "precision_at_k": sum(x.precision_at_k for x in per_query) / n,
        "recall_at_k": sum(x.recall_at_k for x in per_query) / n,
        "mrr": sum(x.mrr for x in per_query) / n,
    }