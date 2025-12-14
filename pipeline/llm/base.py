from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class LLMAdapter(ABC):
    #Abstraktna klasa za LLM, za lakšu promenu između cloud i lokalnog LLM-a.

    @abstractmethod
    def get_model(self) -> Any:  #Pozivanje modela
        raise NotImplementedError

    def ask(self, prompt: str, **kwargs) -> str:   #Komanda koja ide u LLM
        
        model = self.get_model()
        result = model.invoke(prompt, **kwargs)
        
        if hasattr(result, "content"):
            return result.content
        return str(result)
