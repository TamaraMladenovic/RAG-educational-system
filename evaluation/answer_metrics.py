from __future__ import annotations

import os
import json
import csv
import math
import re
from typing import Any, Dict, List, Tuple


from dotenv import load_dotenv
load_dotenv()  # učita .env iz root-a projekta

# -----------------------------
# Utilities
# -----------------------------

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


def load_results_json(app_env: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Loads results.json from common locations.
    Supports your Phase 1 structure:
      {"meta": {...}, "items": [...]}
    Also supports:
      - list[dict]
      - {"results": [...]}, {"data": [...]}, ...
    Returns (path, items_list).
    """
    candidates = [
        f"./evaluation/out/{app_env}/results.json",
        f"./evaluation/out/{app_env}/results_full.json",
        f"./evaluation/out/{app_env}/results_latest.json",
        f"./evaluation/{app_env}_results.json",
        "./evaluation/results.json",
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return path, data

        if isinstance(data, dict):
            for key in ["items", "results", "data", "rows", "records", "evaluations", "outputs"]:
                v = data.get(key)
                if isinstance(v, list):
                    return path, v

            raise ValueError(
                f"Unexpected JSON dict structure in {path}. "
                f"Top-level keys: {list(data.keys())[:50]}"
            )

        raise ValueError(f"Unexpected JSON root type in {path}: {type(data)}")

    raise FileNotFoundError(f"Could not find results.json. Tried: {candidates}")


def pick_answer_field(item: Dict[str, Any]) -> str:
    for k in ["model_answer", "answer", "final_answer", "response"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def pick_gold_field(item: Dict[str, Any]) -> str:
    for k in ["gold_answer", "expected_answer", "reference_answer"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_context_chunks(retrieved_docs: Any, max_chunks: int = 30) -> List[str]:
    chunks: List[str] = []
    if not retrieved_docs:
        return chunks

    if isinstance(retrieved_docs, list):
        for d in retrieved_docs:
            if isinstance(d, str):
                t = d.strip()
                if t:
                    chunks.append(t)
            elif isinstance(d, dict):
                for key in ["text", "content", "page_content", "chunk", "snippet"]:
                    v = d.get(key)
                    if isinstance(v, str) and v.strip():
                        chunks.append(v.strip())
                        break
            if len(chunks) >= max_chunks:
                break

    # de-dup preserve order
    seen = set()
    deduped = []
    for c in chunks:
        sig = c[:500]
        if sig not in seen:
            seen.add(sig)
            deduped.append(c)
    return deduped[:max_chunks]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= 5]


def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0 or nv <= 0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


# -----------------------------
# Embedder (reuse project embedder; fallback to sentence-transformers)
# -----------------------------

class Embedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


def get_embedder() -> Embedder:
    # Try your project adapters
    try:
        from embeddings.local import LocalHFEmbeddingModel  # type: ignore
        model_name = _env("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        e = LocalHFEmbeddingModel(model_name=model_name)

        class _E(Embedder):
            def embed_texts(self, texts: List[str]) -> List[List[float]]:
                return e.embed_documents(texts)

        return _E()
    except Exception:
        pass

    try:
        from pipeline.embeddings.local import LocalHFEmbeddingModel  # type: ignore
        model_name = _env("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        e = LocalHFEmbeddingModel(model_name=model_name)

        class _E(Embedder):
            def embed_texts(self, texts: List[str]) -> List[List[float]]:
                return e.embed_documents(texts)

        return _E()
    except Exception:
        pass

    # Fallback
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model_name = _env("EVAL_EMBEDDING_MODEL", _env("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))
        st = SentenceTransformer(model_name)

        class _E(Embedder):
            def embed_texts(self, texts: List[str]) -> List[List[float]]:
                embs = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return [e.tolist() for e in embs]

        return _E()
    except Exception as ex:
        raise RuntimeError(
            "Could not initialize embedder.\n"
            "Fix: ensure your LocalHFEmbeddingModel is importable OR install sentence-transformers:\n"
            "  pip install sentence-transformers\n"
        ) from ex


# -----------------------------
# Metrics
# -----------------------------

def compute_answer_gold_similarity(embedder: Embedder, answer: str, gold: str) -> float:
    if not answer.strip() or not gold.strip():
        return 0.0
    a_vec, g_vec = embedder.embed_texts([answer, gold])
    return float(cosine(a_vec, g_vec))


def compute_unsupported_sentence_ratio(
    embedder: Embedder,
    answer: str,
    context_chunks: List[str],
    supported_threshold: float = 0.65,
    max_context_chunks: int = 30,
) -> Tuple[float, int, int]:
    sents = split_sentences(answer)
    if not sents:
        return 0.0, 0, 0

    chunks = (context_chunks or [])[:max_context_chunks]
    if not chunks:
        return 1.0, len(sents), len(sents)

    sent_embs = embedder.embed_texts(sents)
    chunk_embs = embedder.embed_texts(chunks)

    unsupported = 0
    for se in sent_embs:
        best = 0.0
        for ce in chunk_embs:
            sim = cosine(se, ce)
            if sim > best:
                best = sim
        if best < supported_threshold:
            unsupported += 1

    ratio = unsupported / max(1, len(sents))
    return float(ratio), unsupported, len(sents)


def quality_label(sim: float, correct_thr: float, partial_thr: float) -> str:
    """
    3-class label:
      - correct: sim >= correct_thr
      - partial: partial_thr <= sim < correct_thr
      - incorrect: sim < partial_thr
    """
    if sim >= correct_thr:
        return "correct"
    if sim >= partial_thr:
        return "partial"
    return "incorrect"


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    app_env = _env("APP_ENV", "local")

    # --- thresholds ---
    CORRECT_THR = float(_env("CORRECT_SIM_THRESHOLD", "0.75"))
    PARTIAL_THR = float(_env("PARTIAL_SIM_THRESHOLD", "0.60"))
    SUPPORTED_SENT_SIM_THRESHOLD = float(_env("SUPPORTED_SENT_SIM_THRESHOLD", "0.65"))
    MAX_CONTEXT_CHUNKS = int(_env("EVAL_MAX_CONTEXT_CHUNKS", "30"))

    if PARTIAL_THR > CORRECT_THR:
        raise ValueError("PARTIAL_SIM_THRESHOLD must be <= CORRECT_SIM_THRESHOLD")

    # optional tag so you can name runs like in Phase 2
    RUN_TAG = _env("RUN_TAG", "")

    path, results = load_results_json(app_env)
    print(f"Loaded: {path} (n={len(results)})")

    embedder = get_embedder()

    out_dir = f"./evaluation/out/{app_env}"
    os.makedirs(out_dir, exist_ok=True)

    # filenames include thresholds so you can compare runs without overwriting
    thr_suffix = f"corr{CORRECT_THR:.2f}_part{PARTIAL_THR:.2f}_supp{SUPPORTED_SENT_SIM_THRESHOLD:.2f}"
    if RUN_TAG:
        thr_suffix = f"{thr_suffix}_{RUN_TAG}"

    items_csv = os.path.join(out_dir, f"answer_items_{thr_suffix}.csv")
    summary_csv = os.path.join(out_dir, f"answer_summary_{thr_suffix}.csv")

    rows: List[Dict[str, Any]] = []

    n = 0
    n_correct = 0
    n_partial = 0
    n_incorrect = 0

    sim_sum = 0.0
    unsup_sum = 0.0

    for item in results:
        qid = item.get("id", "")
        question = (item.get("question") or "").strip()
        gold = pick_gold_field(item)
        answer = pick_answer_field(item)

        retrieved_docs = item.get("retrieved_docs") or item.get("retrieved") or []
        context_chunks = extract_context_chunks(retrieved_docs, max_chunks=MAX_CONTEXT_CHUNKS)

        sim = compute_answer_gold_similarity(embedder, answer, gold)
        label = quality_label(sim, correct_thr=CORRECT_THR, partial_thr=PARTIAL_THR)

        unsupported_ratio, unsupported_cnt, sent_cnt = compute_unsupported_sentence_ratio(
            embedder,
            answer=answer,
            context_chunks=context_chunks,
            supported_threshold=SUPPORTED_SENT_SIM_THRESHOLD,
            max_context_chunks=MAX_CONTEXT_CHUNKS,
        )

        rows.append({
            "id": qid,
            "question": question,
            "answer_gold_similarity": round(sim, 4),
            "quality_label": label,
            "correct_threshold": CORRECT_THR,
            "partial_threshold": PARTIAL_THR,
            "unsupported_sentence_ratio": round(unsupported_ratio, 4),
            "supported_threshold": SUPPORTED_SENT_SIM_THRESHOLD,
            "unsupported_sentences": unsupported_cnt,
            "total_sentences": sent_cnt,
        })

        n += 1
        sim_sum += sim
        unsup_sum += unsupported_ratio

        if label == "correct":
            n_correct += 1
        elif label == "partial":
            n_partial += 1
        else:
            n_incorrect += 1

    # --- write item-level CSV ---
    if rows:
        with open(items_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # --- aggregate summary ---
    correct_rate = n_correct / max(1, n)
    partial_rate = n_partial / max(1, n)
    incorrect_rate = n_incorrect / max(1, n)
    avg_sim = sim_sum / max(1, n)
    avg_unsup = unsup_sum / max(1, n)

    summary_row = {
        "app_env": app_env,
        "n_questions": n,
        "correct_threshold": CORRECT_THR,
        "partial_threshold": PARTIAL_THR,
        "supported_threshold": SUPPORTED_SENT_SIM_THRESHOLD,
        "correct_rate": round(correct_rate, 4),
        "partial_rate": round(partial_rate, 4),
        "incorrect_rate": round(incorrect_rate, 4),
        "avg_answer_gold_similarity": round(avg_sim, 4),
        "avg_unsupported_sentence_ratio": round(avg_unsup, 4),
        "items_csv": os.path.basename(items_csv),
    }

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    # --- console summary ---
    print("\nSUMMARY (3-class)")
    print(f"APP_ENV: {app_env}")
    print(f"Thresholds: correct>={CORRECT_THR}, partial>={PARTIAL_THR} (else incorrect)")
    print(f"Supported sentence threshold: {SUPPORTED_SENT_SIM_THRESHOLD}")
    print(f"Correct:   {n_correct}/{n} ({correct_rate:.3f})")
    print(f"Partial:   {n_partial}/{n} ({partial_rate:.3f})")
    print(f"Incorrect: {n_incorrect}/{n} ({incorrect_rate:.3f})")
    print(f"Avg answer-gold similarity: {avg_sim:.3f}")
    print(f"Avg unsupported_sentence_ratio: {avg_unsup:.3f}")
    print(f"Saved items:   {items_csv}")
    print(f"Saved summary: {summary_csv}")

    # Worst 10 by similarity
    worst = sorted(rows, key=lambda r: r["answer_gold_similarity"])[:10]
    print("\nWORST 10 (by answer_gold_similarity)")
    for r in worst:
        print(f"- id={r['id']} sim={r['answer_gold_similarity']} label={r['quality_label']} unsup_ratio={r['unsupported_sentence_ratio']}")


if __name__ == "__main__":
    main()