from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from pipeline.rag_pipeline import RAGPipeline


# =========================
# CONFIG
# =========================

DATASET_PATH = Path("evaluation/gold_dataset.json")
DEFAULT_TOP_K = 5


# =========================
# HELPERS
# =========================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_docs(raw_docs: Any) -> List[Dict[str, Any]]:
    """
    Normalizuje retrieved dokumente u listu dict-ova.
    """
    if raw_docs is None:
        return []

    if isinstance(raw_docs, list):
        docs = []
        for d in raw_docs:
            if isinstance(d, dict):
                docs.append({
                    "source": d.get("source"),
                    "source_type": d.get("source_type"),
                    "text": d.get("text") or d.get("page_content"),
                    "score": d.get("score")
                })
            else:
                docs.append({
                    "source": None,
                    "source_type": None,
                    "text": str(d),
                    "score": None
                })
        return docs

    return []


def call_rag(rag: RAGPipeline, question: str, top_k: int) -> Dict[str, Any]:
    raw = rag.run(question, top_k=top_k)

    # Ako je string
    if isinstance(raw, str):
        return {"answer": raw, "retrieved_docs": [], "raw_response": raw}

    # Ako je dict
    if isinstance(raw, dict):
        # Probaj standardne ključeve
        answer = (
            raw.get("answer")
            or raw.get("final_answer")
            or raw.get("response")
            or raw.get("result")
            or raw.get("output")
            or raw.get("text")
            or ""
        )

        # Ako i dalje nema, uzmi "prvi string value" iz dict-a (fallback)
        if not answer:
            for v in raw.values():
                if isinstance(v, str) and v.strip():
                    answer = v
                    break

        retrieved = (
            raw.get("retrieved_docs")
            or raw.get("docs")
            or raw.get("documents")
            or raw.get("context_docs")
            or []
        )

        return {
            "answer": answer,
            "retrieved_docs": normalize_docs(retrieved),
            "raw_response": raw
        }

    # fallback
    return {"answer": str(raw), "retrieved_docs": [], "raw_response": raw}


# =========================
# MAIN
# =========================

def main() -> None:

    if not DATASET_PATH.exists():
        raise FileNotFoundError("gold_dataset.json not found in evaluation folder.")

    app_env = os.getenv("APP_ENV", "local")
    top_k = DEFAULT_TOP_K

    # Separate folder for local/cloud
    out_dir = Path("evaluation/out") / app_env
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json(DATASET_PATH)

    rag = RAGPipeline()

    results = {
        "meta": {
            "app_env": app_env,
            "run_time": now_iso(),
            "total_questions": len(dataset),
            "top_k": top_k
        },
        "items": []
    }

    for item in dataset:
        qid = item.get("id")
        question = item.get("question")

        print(f"[{app_env}] Running question {qid}...")

        try:
            output = call_rag(rag, question, top_k)

            results["items"].append({
                "id": qid,
                "question": question,
                "gold_answer": item.get("gold_answer"),
                "evidence": item.get("evidence"),
                "model_answer": output["answer"],
                "retrieved_docs": output["retrieved_docs"],
                "error": None
            })

        except Exception as e:
            results["items"].append({
                "id": qid,
                "question": question,
                "gold_answer": item.get("gold_answer"),
                "evidence": item.get("evidence"),
                "model_answer": "",
                "retrieved_docs": [],
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            })

    save_json(out_dir / "results.json", results)

    print(f"\n✅ Results saved to: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()