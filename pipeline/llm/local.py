from __future__ import annotations
import os
from typing import Any, Optional

from langchain_ollama import ChatOllama

from .base import LLMAdapter


class OllamaQwenAdapter(LLMAdapter):   

    #Lokalni LLM - Qwen2.5:3b-instruct, Ollama server

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        num_ctx: Optional[int] = None,
    ) -> None:
        
        self.model_name = model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b-instruct")

        temp_str = os.getenv("LOCAL_LLM_TEMPERATURE", "0.2")
        self.temperature = temperature if temperature is not None else float(temp_str)

        self.base_url = base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")

        num_ctx_str = os.getenv("LOCAL_LLM_NUM_CTX", "4096")
        self.num_ctx = num_ctx if num_ctx is not None else int(num_ctx_str)

        #ChatOllama instanca
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def get_model(self) -> Any:    #Raw klijent

        return self._llm

    def generate(self, prompt: str) -> str:    #RAG pipeline, prima prompt, vraća str odgovor

        result = self._llm.invoke(prompt)

        content = getattr(result, "content", None)
        if content is not None:
            return content

        return str(result)
