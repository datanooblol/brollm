from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLLM(ABC):
    """Base class for all LLM models"""
    
    @abstractmethod
    def UserMessage(self, text: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def AIMessage(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def SystemMessage(self, text: str) -> Any:
        pass
    
    @abstractmethod
    def run(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        pass
    
    def __call__(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        return self.run(system_prompt, messages)

class BaseEmbedding(ABC):
    """Base class for all embedding models"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

class BaseReranker(ABC):
    """Base class for all reranking models"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        pass