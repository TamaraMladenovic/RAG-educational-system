from __future__ import annotations
from typing import Any

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

    raw = llm.generate(prompt)
    if not raw:  #Ako LLM ne radi, da koristi originalno pitanje
        return question.strip()

    first_line = raw.strip().splitlines()[0].strip()
    if not first_line:
        return question.strip()

    #Skratiti ako vrati predugačak odgovor
    parts = first_line.split()
    if len(parts) > 20:
        first_line = " ".join(parts[:20])

    print(f">>> [REWRITE] Q: {question}")
    print(f">>> [REWRITE] search_query (EN): {first_line}")

    return first_line
