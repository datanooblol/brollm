# brollm

A lightweight, opinionated contract for talking to LLMs. We tell you the shape you should build
in — pack a request, fetch a response, parse it into whatever your program needs — then get out
of your way. Use it exactly as-is, or build on top of it. Either way, you're shipping faster.

## v0.2.0 — a real shift, not a patch

This version changes a lot from the last one, and on purpose. Here's the deal:

- **One contract, not three.** `BaseContract` replaces `BaseLLM`, `BaseEmbedding`, and
  `BaseReranker`. Same shape for chat, embeddings, reranking — whatever you're calling.
- **You're in the driver's seat now.** `input_fn` packs the request and fetches the response —
  context management, retries, the actual provider call, all yours. `output_fn` takes that
  response and turns it into whatever shape your program actually needs — a string, a dataclass
  with token counts, whatever. brollm doesn't guess for you anymore.
- **Providers moved to [`cookbook/`](cookbook/).** AWS Bedrock and Ollama can (and do) change
  their APIs without notice. Instead of brollm quietly breaking under you when that happens,
  provider code lives as copy-paste, version-pinned recipes you control.

Yeah, that's more you have to write than before. That's the point, not a shortcut we took. A
generic base class can't know what *your* response object should look like, what *your* retry
policy should be, or what *your* context window strategy is — you can, and you'll write it once,
correctly, for your actual use case instead of fighting a one-size-fits-none abstraction. Less
magic, more control, and you walk away actually understanding your own LLM layer instead of
trusting a black box. That's a win.

## Features

- **`BaseContract`**: the one shape — `input_fn` in, `output_fn` out. Implement it your way, no
  subclassing gymnastics required.
- **`brollm.retry`**: a `@retry(...)` decorator for wrapping a flaky `input_fn` — exponential or
  fixed backoff, jitter, exception filtering. Stop hand-rolling retry loops.
- **`brollm.parsing.extract_codeblocks`**: pulls fenced codeblocks out of raw LLM text, even if
  the model got cut off mid-stream and never closed the fence.
- **Zero runtime dependencies.** brollm doesn't drag boto3 or httpx into your project just
  because it knows what those libraries are.
- **Provider recipes in [`cookbook/`](cookbook/)**: copy-paste, version-pinned integrations for
  AWS Bedrock and Ollama — a fast start, not a dependency you're locked into.

## Installation

```bash
pip install brollm
```

or

```bash
uv add brollm
```

## Quick Start

Build a contract with your own `input_fn` (pack + fetch) and `output_fn` (parse):

```python
from brollm import BaseContract

def input_fn(system_prompt, messages):
    # own your context management, call whatever provider you want
    return provider_client.call(system_prompt, messages)

def output_fn(response):
    # parse the raw response into whatever your program needs
    return response["text"]

chat = BaseContract(input_fn=input_fn, output_fn=output_fn)
chat("You are a helpful assistant.", [{"role": "user", "content": "Hello!"}])
```

Same shape works for embeddings, reranking, anything — swap `input_fn`/`output_fn` for the call
you're making.

Want token usage, a custom dataclass, parsed codeblocks? That's what `output_fn` is for:

```python
from dataclasses import dataclass
from brollm.parsing import extract_codeblocks

@dataclass
class Tokens:
    input: int
    output: int

@dataclass
class ResponseModel:
    content: str
    usage: Tokens

def output_fn(response):
    return ResponseModel(
        content=response["text"],
        usage=Tokens(input=response["usage"]["input"], output=response["usage"]["output"]),
    )
```

Want retries on a flaky `input_fn`? Wrap it:

```python
from brollm.retry import retry

@retry(max_attempts=3, on=(TimeoutError, ConnectionError))
def input_fn(system_prompt, messages):
    return provider_client.call(system_prompt, messages)
```

For ready-made, version-pinned examples (AWS Bedrock, Ollama) built on this exact pattern, see
[`cookbook/`](cookbook/) — copy the recipe into your project rather than depending on brollm for
it.

## License

MIT License
