# Stack

## Local development

| Concern | Tool | Why |
|---|---|---|
| Package management | Devbox (Nix) | Reproducible environment across machines |
| Python deps | uv | Fast, standard `pyproject.toml`, generates lockfile |
| PostgreSQL + pgvector | Devbox Nix flake | pgvector must be compiled into PostgreSQL; Nix flake handles this |
| LLM inference | llama-cpp-python | llama.cpp bundled as a Python wheel; handles cross-platform GPU automatically |
| Vector store | pgvector | PostgreSQL extension; no separate vector DB needed |
| Graph / RAG logic | LangGraph | Stateful agent graphs with explicit control flow |
| UI | Streamlit | Richer than Chainlit once you need buttons alongside chat |

## Why llama.cpp over Ollama

Ollama wraps llama.cpp but adds a daemon layer that manages model loading behind the scenes — useful for a general-purpose desktop tool, but a source of surprises in a dev environment (unexpected model swaps, hidden context limits). Running llama-server directly gives full control and one fewer moving part.

Both expose an OpenAI-compatible API, so the Python code (`langchain-openai`) is the same either way.

## Why llama-cpp-python instead of the standalone llama.cpp binary

`llama-cpp-python` is a Python binding for llama.cpp that also bundles `llama-server`. It is distributed as platform-specific wheels on a custom index (`abetlen.github.io/llama-cpp-python/whl/`):

- **Linux**: CUDA-enabled wheel (e.g. `cu124`) — no need to build from source or trust a third-party PPA
- **Mac**: Metal-enabled wheel — GPU acceleration via Apple's Metal framework

The alternative — downloading the standalone `llama-server` binary from the llama.cpp GitHub releases — works fine but means managing a separate binary per platform outside of `uv`/`pyproject.toml`. The Python wheel keeps everything in one dependency manager.

The `[server]` extra installs the HTTP server dependencies so `python -m llama_cpp.server` is available. The API it exposes is OpenAI-compatible, identical to the standalone `llama-server`.

**CUDA version compatibility:** Nvidia drivers are backward-compatible. A wheel built for CUDA 12.4 runs on any driver that supports CUDA 12.4 or later. Check your max supported version with `nvidia-smi` (shown top-right as "CUDA Version"). Use the highest wheel version that exists on the index, not necessarily the latest CUDA release.

## AWS deployment (future)

| Concern | Tool | Change from local |
|---|---|---|
| LLM | AWS Bedrock | Swap `ChatOpenAI` → `ChatBedrock` via `LLM_PROVIDER` env var |
| PostgreSQL | RDS PostgreSQL | Same pgvector extension, managed by AWS |
| App hosting | App Runner | Containerised Streamlit, auto-scales |

## Services

Started locally with `devbox services up` via process-compose:

- **postgres** — PostgreSQL 17 with pgvector, data in `.pgdata/`
- **llama-chat** — LLM inference on port 8080, model file in `models/`; built-in chat UI at http://localhost:8080
- **llama-embed** — Embedding inference on port 8081, model file in `models/`
