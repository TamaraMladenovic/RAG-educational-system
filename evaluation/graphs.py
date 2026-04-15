import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_for_system(dir_path: str, system_name: str) -> str | None:
    """
    Find summary JSON by reading the JSON content and matching d["system"].
    This works even if the filename is generic like phase4_summary.json.
    """
    pattern = os.path.join(dir_path, "**", "*.json")
    candidates = glob.glob(pattern, recursive=True)
    candidates = [p for p in candidates if os.path.isfile(p)]

    matches = []

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            if isinstance(d, dict) and d.get("system") == system_name:
                matches.append(path)
        except Exception:
            continue

    if not matches:
        return None

    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def pick_key(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="evaluation/out/phase4_final",
        help="Where to save graphs and final table",
    )
    parser.add_argument(
        "--local_dir",
        default="evaluation/out/local/phase4",
        help="Folder containing local Phase4 outputs",
    )
    parser.add_argument(
        "--cloud_dir",
        default="evaluation/out/cloud/phase4",
        help="Folder containing cloud Phase4 outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    systems = [
        ("RAG_local", find_summary_for_system(args.local_dir, "RAG_local")),
        ("LLMOnly_local", find_summary_for_system(args.local_dir, "LLMOnly_local")),
        ("RAG_cloud", find_summary_for_system(args.cloud_dir, "RAG_cloud")),
        ("LLMOnly_cloud", find_summary_for_system(args.cloud_dir, "LLMOnly_cloud")),
    ]

    print("\n=== FOUND SUMMARY FILES ===")
    for label, path in systems:
        print(label, "->", path)
    print("===========================\n")

    rows = []
    missing = []

    for label, path in systems:
        if not path:
            missing.append(label)
            continue

        d = load_summary(path)

        system_name = pick_key(d, ["system", "System"], default=label)

        gt_sim = to_float(pick_key(d, ["avg_ground_truth_similarity", "avg_gt_similarity"]))
        grounded_avg = to_float(pick_key(d, ["avg_groundedness_score_0_1_2", "avg_groundedness"]))
        fully_grounded = to_float(
            pick_key(d, ["fully_grounded_rate(score=2)", "fully_grounded_rate", "fully_grounded_rate_score_2"])
        )
        no_context = int(to_float(pick_key(d, ["no_context_items", "no_context"]), 0))
        n = int(to_float(pick_key(d, ["n", "total_questions", "N"]), 0))

        rows.append({
            "label": label,
            "system": system_name,
            "n": n,
            "avg_ground_truth_similarity": gt_sim,
            "avg_groundedness_score_0_1_2": grounded_avg,
            "fully_grounded_rate_score_2": fully_grounded,
            "no_context_items": no_context,
            "summary_path": path,
        })

    if missing:
        print("WARNING: Missing summaries for:", ", ".join(missing))

    order = {
        "RAG_local": 0,
        "LLMOnly_local": 1,
        "RAG_cloud": 2,
        "LLMOnly_cloud": 3,
    }
    rows.sort(key=lambda r: order.get(r["label"], 99))

    # --- Save final table as CSV ---
    csv_path = os.path.join(args.out_dir, f"phase4_final_table_{ts}.csv")
    headers = [
        "label",
        "system",
        "n",
        "avg_ground_truth_similarity",
        "avg_groundedness_score_0_1_2",
        "fully_grounded_rate_score_2",
        "no_context_items",
        "summary_path",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print("Saved:", csv_path)

    # --- Prepare plotting data ---
    x_labels = [r["label"] for r in rows]
    gt_vals = [r["avg_ground_truth_similarity"] for r in rows]
    grounded_vals = [r["avg_groundedness_score_0_1_2"] for r in rows]
    fully_vals = [r["fully_grounded_rate_score_2"] for r in rows]

    # Graph 1: GT similarity
    plt.figure(figsize=(9, 5))
    plt.bar(x_labels, gt_vals)
    plt.ylabel("Average Ground Truth Similarity")
    plt.title("Phase 4: Ground Truth Similarity by System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p1 = os.path.join(args.out_dir, f"phase4_gt_similarity_{ts}.png")
    plt.savefig(p1, dpi=200)
    print("Saved:", p1)

    # Graph 2: Groundedness avg
    plt.figure(figsize=(9, 5))
    plt.bar(x_labels, grounded_vals)
    plt.ylabel("Average Groundedness Score (0-2)")
    plt.title("Phase 4: Groundedness by System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, f"phase4_groundedness_avg_{ts}.png")
    plt.savefig(p2, dpi=200)
    print("Saved:", p2)

    # Graph 3: Fully grounded rate
    plt.figure(figsize=(9, 5))
    plt.bar(x_labels, fully_vals)
    plt.ylabel("Fully Grounded Rate (score=2)")
    plt.title("Phase 4: Fully Grounded Rate by System")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p3 = os.path.join(args.out_dir, f"phase4_fully_grounded_rate_{ts}.png")
    plt.savefig(p3, dpi=200)
    print("Saved:", p3)

    plt.show()


if __name__ == "__main__":
    main()