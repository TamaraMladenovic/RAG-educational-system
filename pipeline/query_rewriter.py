from __future__ import annotations
from typing import Any

def _llm_call(llm: Any, prompt: str) -> str:
    if hasattr(llm, "generate") and callable(getattr(llm, "generate")):
        return llm.generate(prompt) or ""
    if hasattr(llm, "invoke") and callable(getattr(llm, "invoke")):
        return llm.invoke(prompt) or ""
    if callable(llm):
        return llm(prompt) or ""
    return ""

#LLM za kreiranje upita na engleskom za klijente za pretraživanje
def rewrite_query_for_search(llm: Any, question: str) -> str:

    prompt = f"""
        You are a search query rewriting assistant.

        YOUR TASK:
        - Read the user's question.
        - Extract the main concept, topic, or entity that should be used
        as a search query for online knowledge sources (Wikipedia, StackOverflow, OpenAlex).
        - Translate that concept to ENGLISH if it is not already in English.
        - Return ONLY a short search phrase or title, WITHOUT any explanation,
        WITHOUT quotes, WITHOUT additional text.

        EXAMPLES:
        Q: When did World War I start?
        A: World War I

        Q: Kada je počeo Prvi svetski rat?
        A: World War I

        Q: Šta je Python kao programski jezik?
        A: Python (programming language)

        Q: Objasni mi ukratko šta je kvantno sprezanje.
        A: quantum entanglement

        Q: What are the main use cases of Redis in web applications?
        A: Redis use cases in web applications

        Now do the same for this question:

        Q: {question}
        A:
        """.strip()

    raw = _llm_call(llm, prompt).strip()
    if not raw:
        return question

    # uzmi samo prvu liniju (LLM ponekad vrati više redova)
    first = raw.splitlines()[0].strip()

    # očisti navodnike/backticks ako ih vrati
    first = first.strip(' "\'`')

    # ✅ SANITY CHECK 1: prekratko = verovatno besmislen odgovor
    if len(first) < 3:
        return question

    # ✅ SANITY CHECK 2: predugačko = skraćujemo na max 12 reči
    if len(first.split()) > 12:
        first = " ".join(first.split()[:12])

    # ✅ dodatni guard: ako je LLM vratio isto što i pitanje ali je pitanje predugačko,
    # makar ga skratimo da pretraga bude stabilnija
    if first.lower() == question.lower() and len(first.split()) > 12:
        first = " ".join(first.split()[:12])
    print(f">>> [REWRITE] Q: {question}")
    print(f">>> [REWRITE] search_query (EN): {first}")

    return first

