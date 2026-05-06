# Models

## Hardware

| Machine | Chip | Memory | Backend |
|---|---|---|---|
| MacBook | Apple M3 Max | 36GB unified RAM | Metal (via llama.cpp `-ngl 99`) |
| Workstation | NVIDIA RTX 3090 | 24GB VRAM | CUDA (via llama.cpp `-ngl 99`) |

## LLM: qwen2.5-32b-instruct (GGUF)

Chosen for both machines — fits comfortably on 36GB (Mac) and 24GB VRAM (RTX 3090).

### Specs

| Property | Value |
|---|---|
| Parameters | 31B (non-embedding) |
| Context window | 128K tokens |
| Max output | 8K tokens |
| Disk size (Q4_K_M GGUF) | ~20GB |
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
# Download GGUF (devbox script)
devbox run model-pull
```

## Embedding model: nomic-embed-text-v1.5 (GGUF)

| Property | Value |
|---|---|
| Size | ~370MB (Q8_0 GGUF) |
| Dimensions | 768 |
| Context window | 8192 tokens |
| Benchmark | Top-tier on MTEB retrieval (open-source) |

llama.cpp serves it on port 8081 with `--embedding --pooling mean`. The pooling mode
matters: `mean` averaging over token embeddings gives the best retrieval quality for
this model (matching how it was trained).

**Important:** switching from snowflake-arctic-embed2 (1024 dims) to nomic-embed-text
(768 dims) changes the vector space — you must re-ingest all documents after the swap.

## Why two llama-server processes?

llama.cpp loads one model per server process. Embedding and generation are separate
models, so they run as `llama-chat` (port 8080) and `llama-embed` (port 8081).
Both expose an OpenAI-compatible API, so the Python code uses `langchain-openai`
with a custom `base_url` — no special Ollama client needed.

## Why llama.cpp over Ollama?

- **Direct control** — flags like `--ctx-size`, `--pooling`, `--batch-size` are
  explicit; Ollama hides them behind a modelfile abstraction
- **No daemon surprises** — Ollama's background process can restart models or
  swap context at unexpected times; llama-server is a single predictable process
- **GGUF-native** — llama.cpp is the reference GGUF runtime; Ollama wraps it
  anyway, adding a layer you don't need

## AWS Bedrock (future deployment)

Replace `ChatOpenAI` with `ChatBedrock` by setting `LLM_PROVIDER=bedrock` in `.env`.
Good model choices on Bedrock:
- `anthropic.claude-3-5-sonnet-20241022-v2:0` — highest quality
- `amazon.nova-pro-v1:0` — cheaper, still good

Embeddings on AWS: swap `OpenAIEmbeddings` for `BedrockEmbeddings` with
`amazon.titan-embed-text-v2:0`. Note: dimensions differ from nomic-embed-text,
so you must re-ingest all documents when switching embedding models.
