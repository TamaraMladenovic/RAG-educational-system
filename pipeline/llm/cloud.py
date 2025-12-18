from __future__ import annotations

import os
from typing import Any, Optional

from groq import Groq, RateLimitError

from .base import LLMAdapter


class GroqLlamaAdapter(LLMAdapter):

    #Cloud - Llama 3.30 70B, radiće preko Groq 

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        
        self.model = model or os.getenv(
            "CLOUD_LLM_MODEL",
            "llama-3.3-70b-versatile",
        )

        temp_str = os.getenv("CLOUD_LLM_TEMPERATURE", "0.2")
        self.temperature = temperature if temperature is not None else float(temp_str)

        max_tokens_str = os.getenv("CLOUD_LLM_MAX_NEW_TOKENS", "1024")
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else int(max_tokens_str)
        )

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY nije setovan u .env / secrets."
            )

        # Inicijalizacija Groq klijenta
        self._client = Groq(api_key=groq_key)

    def get_model(self) -> Any:   #Raw klijent
        
        return self._client

    def generate(self, prompt: str) -> str:   #Prompt se gleda kao user message, poziva se za RAG pipeline

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            msg = completion.choices[0].message
            content = getattr(msg, "content", None)
            if content is not None:
                return content

            return str(msg)

        except RateLimitError:
            return (
                "Privremeno je dostignut limit cloud LLM servisa. "
                "Molim te pokušaj ponovo za minut ili prebaci aplikaciju u lokalni režim."
            )

        except Exception as e:
            # fallback za sve ostalo (network, timeout, itd.)
            return (
                "Došlo je do greške pri generisanju odgovora. "
                "Pokušaj ponovo."
            )

