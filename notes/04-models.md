# Models

## Hardware

| Machine | Chip | Memory | Backend |
|---|---|---|---|
| MacBook | Apple M3 Max | 36GB unified RAM | Metal (via Ollama) |
| Workstation | NVIDIA RTX 3090 | 24GB VRAM | CUDA (via `ollama-cuda`) |

## LLM: qwen2.5:32b

Chosen for both machines — fits comfortably on 36GB (Mac) and 24GB VRAM (RTX 3090).

### Specs

| Property | Value |
|---|---|
| Parameters | 31B (non-embedding) |
| Context window | 128K tokens |
| Max output | 8K tokens |
| Disk size (Ollama, Q4) | ~20GB |
| Languages | 29+ (English, Chinese, French, Spanish, etc.) |
| License | Apache 2.0 |

### Architecture

Uses Grouped Query Attention (GQA) with 40 query heads and 8 key-value heads — this
is why it fits in less memory than a naive 32B model would suggest. GQA reduces the
KV cache size significantly during inference.

### Comparison with alternatives

| Model | Params | Context | Disk (Q4) | Fits RTX 3090 / M3 Max? |
|---|---|---|---|---|
| **qwen2.5:32b** | 31B | 128K | ~20GB | Yes |
| `gemma3:27b` | 27B | 128K | ~17GB | Yes |
| `mistral-small3.1` | 24B | 128K | ~15GB | Yes |
| `llama3.1:8b` | 8B | 128K | ~5GB | Yes (much faster) |
| `llama3.3:70b` | 70B | 128K | ~43GB | No — too large |

`qwen2.5:32b` is the best model that fits on both machines. The only meaningfully
better open-weight option (`llama3.3:70b`) requires ~43GB and won't fit on either.
Among the 30B-class peers, qwen2.5 leads on coding, math, and structured output.

### Why it's good for RAG

- **Strong instruction following** — reliably stays within the retrieved context
  and doesn't hallucinate beyond it
- **Structured output** — excellent JSON generation, useful when the graph needs
  structured responses
- **Long context** — 128K window means you can stuff many retrieved chunks into
  the prompt without worrying about truncation
- **Reasoning** — handles multi-hop questions well (e.g. synthesising across
  several retrieved passages)

```bash
ollama pull qwen2.5:32b
```

On the RTX 3090 machine, install `ollama-cuda` instead of `ollama` in `devbox.json`.

## Embedding model: snowflake-arctic-embed2

| Property | Value |
|---|---|
| Size | ~1.2GB |
| Dimensions | 1024 |
| Context window | 8192 tokens |
| Benchmark | Top-tier on MTEB retrieval |

The large context window means chunks can be kept large without truncation.
Embedding models are much smaller than LLMs — hardware barely matters for them.

```bash
ollama pull snowflake-arctic-embed2
```

## Why separate embedding and LLM models?

- The embedding model runs at both ingest time and query time
- The LLM only runs at query time
- Embedding models are optimised for similarity, not generation
- Using the same embedding model for ingestion and retrieval is required —
  mismatched models produce incompatible vector spaces

## AWS Bedrock (future deployment)

Replace `ChatOllama` with `ChatBedrock` by setting `LLM_PROVIDER=bedrock` in `.env`.
Good model choices on Bedrock:
- `anthropic.claude-3-5-sonnet-20241022-v2:0` — highest quality
- `amazon.nova-pro-v1:0` — cheaper, still good

Embeddings on AWS: swap `OllamaEmbeddings` for `BedrockEmbeddings` with
`amazon.titan-embed-text-v2:0`. Note: dimensions differ from snowflake-arctic-embed2,
so you must re-ingest all documents when switching embedding models.
