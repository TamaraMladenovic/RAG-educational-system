import csv
from pathlib import Path

INPUT = Path("evaluation/out/local/answer_items_corr0.75_part0.60_supp0.65.csv")

rows = []

with open(INPUT, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["answer_gold_similarity"] = float(r["answer_gold_similarity"])
        rows.append(r)

# sortiranje po similarity
rows_sorted = sorted(rows, key=lambda x: x["answer_gold_similarity"])

# uzorak
worst = rows_sorted[:5]
best = rows_sorted[-5:]
middle = rows_sorted[len(rows)//2-2 : len(rows)//2+3]

sample = worst + middle + best

print("\nSELECTED QUESTIONS\n")

for r in sample:
    print(
        f"id={r['id']} | sim={r['answer_gold_similarity']} | label={r['quality_label']} | question={r['question']}"
    )