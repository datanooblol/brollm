# brollm

A lightweight Python library providing unified interfaces for LLM models, embeddings, and rerankers. Built for AI agent development with consistent APIs across different providers.

## Features

- **Unified Interface**: Same API across AWS Bedrock and Ollama
- **LLM Models**: Chat completion with system prompts and message history
- **Embeddings**: Text embedding generation for semantic search
- **Rerankers**: Document reranking capabilities (base class ready)
- **Lightweight**: Minimal dependencies, maximum flexibility

## Installation

```bash
pip install brollm
```

or

```bash
uv add brollm
```

## Quick Start

### AWS Bedrock

```python
from brollm import BedrockChat, BedrockEmbedding

# Chat completion
chat = BedrockChat(model_name="us.meta.llama3-2-11b-instruct-v1:0")
messages = [
    chat.UserMessage("What is machine learning?"),
    chat.AIMessage("Machine learning is..."),
    chat.UserMessage("Give me an example")
]
response = chat.run("You are a helpful AI assistant", messages)
print(response)

# Embeddings
embedding = BedrockEmbedding()
vector = embedding.embed_text("Hello world")
print(f"Embedding dimension: {len(vector)}")
```

### Ollama

```python
from brollm import OllamaChat, OllamaEmbedding

# Chat completion
chat = OllamaChat(model_name="qwen3:8b")
messages = [
    chat.UserMessage("Explain quantum computing"),
    chat.AIMessage("Quantum computing uses..."),
    chat.UserMessage("What are the applications?")
]
response = chat.run("You are a physics expert", messages)
print(response)

# Embeddings
embedding = OllamaEmbedding(model_name="nomic-embed-text")
vectors = embedding.embed_texts(["Hello", "World", "AI"])
print(f"Generated {len(vectors)} embeddings")
```

### Multimodal Support (Bedrock)

```python
from brollm import BedrockChat

chat = BedrockChat()
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

messages = [
    chat.UserMessage("Describe this image", image_bytes=image_bytes, image_format="jpeg")
]
response = chat.run("You are a vision AI assistant", messages)
print(response)
```

## Provider Switching

Switch between providers without changing your code structure:

```python
# Use Bedrock
llm = BedrockChat(temperature=0.7)

# Switch to Ollama
llm = OllamaChat(temperature=0.7)

# Same interface for both
messages = [llm.UserMessage("Hello AI!")]
response = llm.run("You are helpful", messages)
```

## Configuration

### Bedrock
```python
chat = BedrockChat(
    model_name="us.meta.llama3-2-11b-instruct-v1:0",
    temperature=0.7,
    region_name="us-west-2",
    aws_access_key_id="your-key",  # Optional, uses default AWS config
    aws_secret_access_key="your-secret",  # Optional
    aws_session_token="your-token"  # Optional
)
```

### Ollama
```python
chat = OllamaChat(
    model_name="qwen3:8b",
    temperature=0.7,
    base_url="http://localhost:11434"  # Default Ollama endpoint
)
```

## License

MIT License
