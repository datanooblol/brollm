import asyncio
import httpx
from typing import List

async def ollama_chat_async(prompt: str, model: str = "qwen3:8b", base_url: str = "http://localhost:11434") -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f'{base_url}/api/chat', json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })
        return response.json()['message']['content']

# Usage example
async def main():
    # Three concurrent chats
    prompts = [
        "What is Python?",
        "Explain machine learning",
        "Tell me about async programming"
    ]
    
    tasks = [ollama_chat_async(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"Chat {i}: {result[:100]}...\n")
    
if __name__ == "__main__":
    asyncio.run(main())