from typing import Any, Dict, List
import requests
from .base import BaseLLM, BaseEmbedding

class OllamaChat(BaseLLM):
    def __init__(
            self, 
            model_name: str = "qwen3:8b",
            temperature: float = 0,
            base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url

    def UserMessage(self, text: str, **kwargs) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def AIMessage(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": text}

    def SystemMessage(self, text: str) -> Dict[str, Any]:
        return {"role": "system", "content": text}

    def run(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        all_messages = [self.SystemMessage(system_prompt)] + messages
        
        response = requests.post(f'{self.base_url}/api/chat', json={
            "model": self.model_name,
            "messages": all_messages,
            "stream": False,
            "options": {"temperature": self.temperature}
        })
        
        return response.json()['message']['content']

class OllamaEmbedding(BaseEmbedding):
    def __init__(
            self,
            model_name: str = "nomic-embed-text",
            base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def embed_text(self, text: str) -> List[float]:
        response = requests.post(f'{self.base_url}/api/embeddings', json={
            "model": self.model_name,
            "prompt": text
        })
        return response.json()['embedding']

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]