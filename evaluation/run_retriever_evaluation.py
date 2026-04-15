from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import csv
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # učita .env iz root-a projekta

# ⬇ APSOLUTNI IMPORT (bez relative)
from evaluation.retriever_metrics import aggregate, compute_metrics_for_item


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    # ✅ Dinamički APP_ENV (local ili cloud)
    app_env = os.getenv("APP_ENV", "local")

    # Folder se bira automatski
    out_dir = Path("evaluation/out") / app_env
    results_path = out_dir / "results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    data = load_json(results_path)

    items: List[Dict[str, Any]] = data.get("items", [])
    meta: Dict[str, Any] = data.get("meta", {})

    # Uzima top_k iz Phase 1 meta
    k = int(os.getenv("EVAL_TOP_K", str(meta.get("top_k", 5))))
    threshold = float(os.getenv("EVIDENCE_MATCH_THRESHOLD", "0.12"))

    per_query = []
    per_question_rows = []

    for it in items:
        retrieved_docs = it.get("retrieved_docs") or []
        evidence = it.get("evidence")
        err = it.get("error")

        m = compute_metrics_for_item(
            retrieved_docs=retrieved_docs,
            evidence=evidence,
            k=k,
            threshold=threshold,
        )

        per_query.append(m)

        per_question_rows.append(
            {
                "id": it.get("id"),
                "question": it.get("question"),
                "precision_at_k": m.precision_at_k,
                "recall_at_k": m.recall_at_k,
                "mrr": m.mrr,
                "first_relevant_rank": m.first_relevant_rank,
                "relevant_in_top_k": m.relevant_in_top_k,
                "k": m.k,
                "had_error": err is not None,
            }
        )

    summary = aggregate(per_query)

    payload = {
        "meta": {
            "app_env": app_env,
            "source_results": str(results_path),
            "k": k,
            "threshold": threshold,
            "n_questions": len(items),
            "n_errors": sum(1 for it in items if it.get("error") is not None),
            "phase1_meta": meta,
        },
        "summary": summary,
        "per_question": per_question_rows,
    }

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"k{k}_thr{threshold}"

    # =========================
    # SAVE JSON
    # =========================
    json_path = runs_dir / f"retriever_metrics_{base_name}.json"
    save_json(json_path, payload)

    latest_json = out_dir / "retriever_metrics_latest.json"
    save_json(latest_json, payload)

    # =========================
    # SAVE SUMMARY CSV
    # =========================
    summary_csv_path = runs_dir / f"summary_{base_name}.csv"

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "app_env",
            "k",
            "threshold",
            "n_questions",
            "precision_at_k",
            "recall_at_k",
            "mrr",
        ])
        writer.writerow([
            app_env,
            k,
            threshold,
            len(items),
            summary["precision_at_k"],
            summary["recall_at_k"],
            summary["mrr"],
        ])

    # =========================
    # SAVE PER QUESTION CSV
    # =========================
    per_question_csv_path = runs_dir / f"per_question_{base_name}.csv"

    with open(per_question_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "question",
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "first_relevant_rank",
            "relevant_in_top_k",
            "had_error",
        ])

        for row in per_question_rows:
            writer.writerow([
                row["id"],
                row["question"],
                row["precision_at_k"],
                row["recall_at_k"],
                row["mrr"],
                row["first_relevant_rank"],
                row["relevant_in_top_k"],
                row["had_error"],
            ])

    print(f"\n✅ Saved JSON: {json_path}")
    print(f"✅ Saved Summary CSV: {summary_csv_path}")
    print(f"✅ Saved Per-Question CSV: {per_question_csv_path}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()