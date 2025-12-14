from __future__ import annotations
import os

from .base import LLMAdapter
from .local import OllamaQwenAdapter
from .cloud import GroqLlamaAdapter


def get_llm_adapter(env: str | None = None) -> LLMAdapter:    
    
    #Na osnovu .env gleda koji LLM koristi (cloud/local)

    env = env or os.getenv("APP_ENV", "local").lower()

    if env == "local":
        return OllamaQwenAdapter()
    elif env == "cloud":
        return GroqLlamaAdapter()
    else:
        raise ValueError(f"Unknown APP_ENV='{env}', koristi 'local' ili 'cloud'.")
