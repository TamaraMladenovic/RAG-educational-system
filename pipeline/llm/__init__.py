# src/llm_adapters/__init__.py
from .base import LLMAdapter
from .factory import get_llm_adapter

__all__ = ["LLMAdapter", "get_llm_adapter"]
