# Cookbook recipe — tested with httpx==0.28.1, Ollama /api/chat and /api/embed
#
# Copy this file into your project and adapt it as needed. Ollama's local API
# can change without notice, so this recipe is not shipped as part of the
# installable `brollm` package — re-verify against this recipe when you
# upgrade your Ollama server or httpx version.
#
# Built on brollm's BaseContract: `input_fn` packs messages and calls
# Ollama's HTTP API (wrapped with brollm's `retry` since a local server can
# be slow to warm up), `output_fn` parses the raw response into plain text
# (chat) or a vector/list of vectors (embedding). Swap `output_fn` for your
# own parsing without touching `input_fn`.
#
#   chat = make_ollama_chat()
#   messages = [user_message("Hello!")]
#   chat("You are a helpful assistant.", messages)  # -> str
#
#   embed = make_ollama_embedding()
#   embed("hello world")  # -> list[float]
#   embed(["hello", "world"])  # -> list[list[float]]

from typing import Any

import httpx

from brollm import BaseContract
from brollm.retry import retry


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def ai_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def make_ollama_chat(
        model_name: str = "qwen3:8b",
        temperature: float = 0,
        base_url: str = "http://localhost:11434",
        max_attempts: int = 3,
) -> BaseContract:

    @retry(max_attempts=max_attempts, on=(httpx.TransportError,))
    def input_fn(system_prompt: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        response = httpx.post(f"{base_url}/api/chat", json={
            "model": model_name,
            "messages": all_messages,
            "stream": False,
            "options": {"temperature": temperature},
        })
        response.raise_for_status()
        return response.json()

    def output_fn(response: dict[str, Any]) -> str:
        return response["message"]["content"]

    return BaseContract(input_fn=input_fn, output_fn=output_fn)


def make_ollama_embedding(
        model_name: str = "nomic-embed-text:v1.5",
        base_url: str = "http://localhost:11434",
        max_attempts: int = 3,
) -> BaseContract:

    @retry(max_attempts=max_attempts, on=(httpx.TransportError,))
    def input_fn(text_or_texts: str | list[str]) -> dict[str, Any]:
        response = httpx.post(f"{base_url}/api/embed", json={
            "model": model_name,
            "input": text_or_texts,
        })
        response.raise_for_status()
        return {"batched": isinstance(text_or_texts, list), **response.json()}

    def output_fn(response: dict[str, Any]) -> list[float] | list[list[float]]:
        embeddings = response["embeddings"]
        return embeddings if response["batched"] else embeddings[0]

    return BaseContract(input_fn=input_fn, output_fn=output_fn)
