# Cookbook

`brollm` ships only the base contract (`BaseContract`, `input_fn`/`output_fn`) plus small parsing
and retry utilities. Provider integrations (AWS Bedrock, Ollama, ...) live here as copy-paste
recipes instead of being part of the installed package.

Why: provider SDKs and local APIs (boto3/Bedrock, Ollama's HTTP API, etc.) can change without
notice. Shipping them inside `brollm` means every breaking upstream change breaks everyone's
pinned `brollm` version. Keeping them as recipes means you copy the file into your own project,
pin the dependency versions noted at the top of the file, and control when you take an upstream
upgrade.

Each recipe is a factory function that builds a `BaseContract`: `input_fn` packs the call and
talks to the provider (wrapped with brollm's `retry` where it makes sense — throttling, flaky
local servers), `output_fn` parses the raw response into plain text or vectors. Swap `output_fn`
for your own parsing (e.g. `brollm.parsing.extract_codeblocks`) without touching `input_fn`.

Recipes are plain `.py` files, not notebooks — the whole point is that you copy the file into
your project as real, importable, lintable source. If you want a narrated walkthrough of a
recipe running end-to-end, put that in `notebooks/` instead; keep the recipe itself as `.py`.

## Usage

1. Copy the recipe file (e.g. `bedrock/bedrock_chat.py`) into your project.
2. Install the dependency versions noted in the file's header comment.
3. Adapt as needed — these are starting points, not a stable API surface.

## Recipes

- `bedrock/bedrock_chat.py` — `make_bedrock_chat(...)`, AWS Bedrock chat as a `BaseContract`.
- `ollama/ollama_chat.py` — `make_ollama_chat(...)` and `make_ollama_embedding(...)`, Ollama chat
  and embeddings as `BaseContract`s.
