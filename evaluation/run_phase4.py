# evaluation/run_phase4.py
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()  # učita .env iz root-a projekta

# -----------------------------
# Robust IO
# -----------------------------
def load_results_json(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Supports:
    1) {"meta": {...}, "items": [ {...}, ... ]}
    2) [ {...}, ... ]
    3) single dict {...}
    4) legacy: list[str] where each str is a JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # (1) meta+items
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"], data.get("meta", {})

    # (2) list
    if isinstance(data, list):
        if not data:
            return [], {}
        # (4) list of JSON strings
        if isinstance(data[0], str):
            fixed: List[Dict[str, Any]] = []
            for x in data:
                try:
                    obj = json.loads(x)
                    if isinstance(obj, dict):
                        fixed.append(obj)
                except Exception:
                    pass
            return fixed, {}
        # list of dicts
        return data, {}

    # (3) single dict
    if isinstance(data, dict):
        return [data], {}

    raise ValueError(f"Unsupported JSON structure in {path}: {type(data)}")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Field extraction
# -----------------------------
def normalize_ws(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_answer(item: Dict[str, Any]) -> str:
    for k in ["model_answer", "answer", "final_answer", "response"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def pick_gold(item: Dict[str, Any]) -> str:
    for k in ["gold_answer", "reference_answer", "expected_answer"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def extract_contexts(item: Dict[str, Any], max_chars: int = 12000) -> List[str]:
    """
    Extract contexts from retrieved_docs (list[dict] or list[str]) or fallback context fields.
    Returns a list (usually one big joined string) for simplicity.
    """
    ctxs: List[str] = []

    rdocs = item.get("retrieved_docs")

    if isinstance(rdocs, list):
        for d in rdocs:
            if isinstance(d, str):
                t = d
            elif isinstance(d, dict):
                t = (
                    d.get("text")
                    or d.get("content")
                    or d.get("chunk")
                    or d.get("page_content")
                    or ""
                )
            else:
                t = ""
            t = normalize_ws(str(t))
            if t:
                ctxs.append(t)

    if not ctxs:
        # fallback fields if you ever store a pre-joined context
        for k in ["final_context", "context", "contexts"]:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                ctxs = [normalize_ws(v)]
                break
            if isinstance(v, list) and v:
                ctxs = [normalize_ws(str(x)) for x in v if str(x).strip()]
                break

    if ctxs:
        joined = "\n".join(ctxs)[:max_chars]
        return [joined]

    return []


# -----------------------------
# Metric 1: Ground-truth similarity (0..1)
# -----------------------------
def ground_truth_similarity(answer: str, gold: str) -> float:
    a = normalize_ws(answer).lower()
    g = normalize_ws(gold).lower()
    if not a or not g:
        return 0.0
    return SequenceMatcher(None, a, g).ratio()


# -----------------------------
# Metric 2: Groundedness heuristic (0/1/2)
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
_WORD = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)


def sentence_claims(answer: str, max_claims: int = 8) -> List[str]:
    sents = _SENT_SPLIT.split(normalize_ws(answer))
    sents = [s.strip() for s in sents if len(s.strip()) >= 12]
    if not sents:
        sents = [normalize_ws(answer)]
    return sents[:max_claims]


def keyword_set(text: str) -> set:
    return set(w.lower() for w in _WORD.findall(text) if len(w) >= 4)


def groundedness_heuristic(answer: str, contexts: List[str]) -> Tuple[int, float]:
    """
    Returns (score_0_1_2, support_ratio).
    - Extract up to N claims (sentences) from answer.
    - A claim is 'supported' if keyword overlap with context crosses a conservative threshold.
    """
    ans = normalize_ws(answer)
    if not ans:
        return 0, 0.0

    ctx = normalize_ws("\n".join(contexts)) if contexts else ""
    if not ctx:
        return 0, 0.0  # no context => can't be grounded

    ctx_kw = keyword_set(ctx)
    claims = sentence_claims(ans)

    supported = 0
    for c in claims:
        c_kw = keyword_set(c)
        if not c_kw:
            continue
        overlap = len(c_kw & ctx_kw)
        # conservative threshold: at least 2 words and ~25% of claim keywords
        if overlap >= max(2, int(0.25 * len(c_kw))):
            supported += 1

    ratio = supported / max(1, len(claims))

    if ratio >= 0.66:
        return 2, ratio
    if ratio >= 0.33:
        return 1, ratio
    return 0, ratio


# -----------------------------
# Output row
# -----------------------------
@dataclass
class Row:
    id: Any
    question: str
    system: str
    gt_sim: float
    groundedness: int
    support_ratio: float
    has_context: int


def make_safe_filename(s: str) -> str:
    """
    Make a safe string for filenames (keeps letters, numbers, _, -).
    """
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "system"


def main() -> None:
    app_env = os.getenv("APP_ENV", "local")

    # input
    in_path = os.getenv("PHASE4_INPUT", f"evaluation/out/{app_env}/results.json")

    # output dir
    out_dir = os.getenv("PHASE4_OUTDIR", f"evaluation/out/{app_env}/phase4")

    # system label (IMPORTANT: set this per run)
    system_name = os.getenv("PHASE4_SYSTEM", f"RAG_{app_env}")

    ensure_dir(out_dir)

    items, meta = load_results_json(in_path)

    rows: List[Row] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        _id = item.get("id")
        q = normalize_ws(str(item.get("question", "")))

        ans = pick_answer(item)
        gold = pick_gold(item)
        ctxs = extract_contexts(item)

        gt = ground_truth_similarity(ans, gold)
        gscore, ratio = groundedness_heuristic(ans, ctxs)

        rows.append(
            Row(
                id=_id,
                question=q,
                system=system_name,
                gt_sim=round(gt, 4),
                groundedness=int(gscore),
                support_ratio=round(ratio, 4),
                has_context=1 if ctxs and ctxs[0].strip() else 0,
            )
        )

    # -----------------------------
    # Output files (per system) ✅ no overwrite
    # -----------------------------
    safe_system = make_safe_filename(system_name)

    csv_path = os.path.join(out_dir, f"phase4_{safe_system}.csv")
    summary_path = os.path.join(out_dir, f"phase4_{safe_system}_summary.json")

    # Save CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "system", "gt_sim", "groundedness", "support_ratio", "has_context"])
        for r in rows:
            w.writerow([r.id, r.question, r.system, r.gt_sim, r.groundedness, r.support_ratio, r.has_context])

    # Summary
    n = len(rows)
    avg_gt = sum(r.gt_sim for r in rows) / max(1, n)
    avg_ground = sum(r.groundedness for r in rows) / max(1, n)
    fully_grounded_rate = sum(1 for r in rows if r.groundedness == 2) / max(1, n)
    noctx = sum(1 for r in rows if r.has_context == 0)

    summary = {
        "app_env": app_env,
        "system": system_name,
        "n": n,
        "avg_ground_truth_similarity": round(avg_gt, 4),
        "avg_groundedness_score_0_1_2": round(avg_ground, 4),
        "fully_grounded_rate(score=2)": round(fully_grounded_rate, 4),
        "no_context_items": int(noctx),
        "input_path": in_path,
        "output_csv": csv_path,
        "input_meta": meta,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Phase 4 auto done")
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()