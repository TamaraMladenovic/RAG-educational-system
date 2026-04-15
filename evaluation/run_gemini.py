from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
from google import genai  # new SDK

from dotenv import load_dotenv
load_dotenv()  # učita .env iz root-a projekta

MODEL = "gemini-2.0-flash"  # primer iz zvanične migracije :contentReference[oaicite:1]{index=1}

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    client = genai.Client(api_key=api_key)  # :contentReference[oaicite:2]{index=2}

    gold_path = Path("evaluation/gold_dataset.json")
    out_dir = Path("evaluation/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = out_dir / f"gemini_results_{ts}.json"

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    items = gold["items"] if isinstance(gold, dict) and "items" in gold else gold

    results = {"meta": {"system": "Gemini", "model": MODEL, "generated_at": ts, "n_items": len(items)}, "items": []}

    for it in items:
        qid = it["id"]
        question = it["question"]

        resp = client.models.generate_content(
            model=MODEL,
            contents=question
        )  # :contentReference[oaicite:3]{index=3}

        results["items"].append({
            "id": qid,
            "question": question,
            "model_answer": getattr(resp, "text", None) or "",
            "error": None,
        })

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()