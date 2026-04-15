# evaluation/run_llm_only_baseline.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()  # učita .env iz root-a projekta

# ✅ Koristi tvoj postojeći adapter / pipeline import
# U većini tvojih modula ovo već postoji:
# - pipeline.llm_adapters (npr. get_llm())
# - ili RAGPipeline().llm
#
# Ja pravim "najbezbolniji" pristup:
# 1) probaj get_llm() ako postoji
# 2) fallback: RAGPipeline().llm

def load_results_json(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"], data.get("meta", {})
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        return [data], {}
    raise ValueError("Unsupported JSON structure")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def get_llm():
    # 1) pokušaj tvoj helper (ako postoji)
    try:
        from pipeline.llm_adapters import get_llm  # type: ignore
        return get_llm()
    except Exception:
        pass

    # 2) fallback: uzmi llm iz RAGPipeline (ali NE zovi retrieval)
    from pipeline.rag_pipeline import RAGPipeline  # type: ignore
    rag = RAGPipeline()
    # pretpostavka: rag.llm postoji i ima .generate(prompt)
    return rag.llm

def build_prompt(question: str) -> str:
    # Minimalan prompt za LLM-only baseline
    return (
        "Answer the question clearly and correctly.\n"
        "If you are not sure, say you are not sure.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def llm_generate(llm, prompt: str) -> str:
    # pokušaj generički interfejs
    if hasattr(llm, "generate"):
        return llm.generate(prompt)
    if callable(llm):
        return llm(prompt)
    raise RuntimeError("LLM object has no supported generate interface")

def main():
    app_env = os.getenv("APP_ENV", "local")
    in_path = f"evaluation/out/{app_env}/results.json"
    out_dir = f"evaluation/out/{app_env}/llm_only"
    out_path = f"{out_dir}/results.json"

    items, meta = load_results_json(in_path)
    ensure_dir(out_dir)

    llm = get_llm()

    out_items: List[Dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        q = (it.get("question") or "").strip()
        if not q:
            continue

        prompt = build_prompt(q)
        try:
            ans = llm_generate(llm, prompt)
        except Exception as e:
            ans = ""
            it = dict(it)
            it["error"] = f"llm_only_error: {e}"

        new_it = dict(it)
        new_it["model_answer"] = ans
        # VAŽNO: LLM-only nema retrieved docs (ili ostavi prazno)
        new_it["retrieved_docs"] = []
        out_items.append(new_it)

        # mali throttling da ne udari rate limit na cloud-u
        time.sleep(float(os.getenv("LLM_ONLY_SLEEP", "0.2")))

        if i % 10 == 0:
            print(f"[LLM-ONLY] done {i}/{len(items)}")

    out_obj = {
        "meta": {
            **(meta or {}),
            "app_env": app_env,
            "baseline": "llm_only",
            "run_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_questions": len(out_items),
        },
        "items": out_items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("✅ LLM-only baseline saved to:", out_path)

if __name__ == "__main__":
    main()