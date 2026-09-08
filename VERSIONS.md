# Version History

## 0.2.0
- Rebuilt around one contract: `BaseContract(input_fn, output_fn)`. `BaseLLM`, `BaseEmbedding`,
  and `BaseReranker` are gone — one shape now covers chat, embeddings, and reranking alike.
- `input_fn` packs the request and fetches the raw response, `output_fn` parses it into
  whatever your program needs. You own context management and response shape completely —
  that's the trade: less structure handed to you, more control in your hands.
- Provider integrations (Bedrock, Ollama) moved out of the installable package and into
  `cookbook/` as copy-paste, version-pinned recipes. `brollm` itself is back to zero runtime
  dependencies.
- Added `brollm.retry` — a `@retry(...)` decorator for wrapping a flaky `input_fn`
  (exponential/fixed backoff, jitter, exception filtering).
- Added `brollm.parsing.extract_codeblocks` — pulls fenced codeblocks (closed or truncated
  mid-stream) out of raw LLM text output.

## 0.1.2
- Update base.py and ollama.py with embedding

## 0.1.1
- Initialize with bedrock
