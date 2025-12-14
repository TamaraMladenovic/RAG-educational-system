from __future__ import annotations
from typing import List, Dict, Any


def reclean_text(text: str) -> str:    #Reformatiranje chunkova ukoliko je potrebno (višestrucci razmaci i prazni redovi)

    if not text:
        return ""

    text = text.strip()
    lines = [line.strip() for line in text.split("\n")]
    lines = [l for l in lines if l not in ("", " ", "\t")]
    return "\n".join(lines)


def format_single_chunk(chunk: Dict[str, Any], idx: int) -> str:   #Formatiranje jednog chunka

    text = reclean_text(chunk.get("text", ""))
    source = chunk.get("source", "unknown").upper()

    return f"""
### Chunk {idx} — Source: {source}
{text}
""".strip()


def format_context_block(chunks: List[Dict[str, Any]]) -> str:    #Spaja chunkove u kontekst

    if not chunks:
        return "NO CONTEXT FOUND."

    formatted = [format_single_chunk(c, i + 1) for i, c in enumerate(chunks)]
    return "\n\n".join(formatted)


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:   #LLM prompt

    print(">>> [DEBUG] build_prompt POZVAN, broj chunkova:", len(chunks))
    
    context_block = format_context_block(chunks)

    prompt = f"""
        You are an educational AI assistant with access to retrieved knowledge.
        The context you recieve may be on another language, 
        but you MUST unerstand that text and you MUST reply in the same langueage the user question is.

        USER QUESTION:
        {question}

        RELEVANT CONTEXT (do not hallucinate; use ONLY what is provided):
        {context_block}

        INSTRUCTIONS:
        - Answer the user's question using ONLY the context above.
        - If context is insufficient, say "Not enough information in retrieved sources."
        - Be precise, structured, and concise.
        """

    return prompt.strip()
