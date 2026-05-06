# Models

## Hardware

| Machine | Chip | Memory | Backend |
|---|---|---|---|
| MacBook | Apple M3 Max | 36GB unified RAM | Metal (via llama.cpp `-ngl 99`) |
| Workstation | NVIDIA RTX 3090 | 24GB VRAM | CUDA (via llama.cpp `-ngl 99`) |

## LLM: Qwen3.6-27B (abliterated, GGUF)

Qwen3.6 is a multimodal model family from Alibaba released in 2025. "3.6" is a
generation name, not a parameter count — like naming a car model year. The 27B
variant is the dense (non-MoE) model in the family.

We use the abliterated variant from `huihui-ai`, quantized to Q4_K_M by Abiray.

### Specs

| Property | Value |
|---|---|
| Parameters | 27B |
| Architecture | Dense transformer with GQA |
| Layers | 64 |
| KV heads | 4 (query heads: 24) |
| Head dimension | 256 |
| Disk size (Q4_K_M GGUF) | ~15.4 GB |
| License | Apache 2.0 |

### KV cache and context size

The KV cache is the memory that stores intermediate attention state during
generation. It grows with context length — the longer the conversation or the
more retrieved chunks in the prompt, the more memory it needs.

**Formula:**
```
KV cache per token = 2 × KV_heads × head_dim × layers × bytes_per_element
                   = 2 × 4 × 256 × 64 × 2 bytes (fp16)
                   = 256 KB per token
```

With 15.4 GB used by the model weights and 24 GB VRAM total, there's ~8.6 GB
headroom. At fp16 that gives ~34K tokens of context. But llama.cpp can quantize
the KV cache too:

| KV cache type | Bytes/element | Context at 8.6 GB headroom |
|---|---|---|
| fp16 (default) | 2 | ~34K tokens |
| q8_0 | 1 | ~68K tokens |
| q4_0 | 0.5 | ~136K tokens |

We use `-ctk q8_0 -ctv q8_0` (set in `process-compose.yaml`), targeting
`--ctx-size 49152` (48K) — a safe margin below the 68K theoretical max.

**What does 48K tokens mean in practice?**

- ~36,000 words
- ~120 paperback novel pages (at ~300 words/page)
- About half a full-length novel, or one complete Sherlock Holmes novel

**What `-ctk` and `-ctv` mean:**
- `ctk` = cache-type-k = quantization applied to the **Key** half of KV cache
- `ctv` = cache-type-v = quantization applied to the **Value** half
- `q8_0` = 8 bits per element, variant 0 (simple linear quant — minimal quality loss)
- `q4_0` = 4 bits per element (half the memory, slight quality loss)

The `_0` suffix means "variant 0" — basic linear quantization. Unlike model weight
quants (which use fancier schemes like `q4_k_m`), KV cache quants only use the
simpler `_0` variants because the activations are transient and re-quantization
overhead matters more at inference time.

### GGUF quantization variants explained

GGUF is the file format llama.cpp uses for model weights. Quantization reduces
precision to save memory — a tradeoff between size and quality.

**Name anatomy: `Q4_K_M`**
- `Q4` — 4 bits per weight element (vs 16 bits for fp16, halving memory use)
- `K` — "K-quant": a smarter scheme that groups weights and assigns shared scale
  factors, preserving more information than naive rounding
- `M` — medium variant (S = small, M = medium, L = large within the same bit level)

**Importance-weighted quants (`IQ`):**
`IQ4_XS`, `IQ4_NL` — assign more bits to weights that matter most for output
quality, fewer bits to weights that tolerate loss. Better quality per GB than
standard Q4, but slower to load.

**Unsloth Dynamic (`UD`) quants:**
`UD-Q4_K_XL` etc. — same idea applied at the layer level: layers earlier and
later in the network (which affect output more) get more bits; middle layers get
fewer. Generally the best quality for a given file size.

**Available abliterated variants** (`Abiray/Huihui-Qwen3.6-27B-abliterated-GGUF`):

| File | Size | Notes |
|---|---|---|
| `Q3_K_M` | 12.4 GB | 3-bit — noticeable quality drop |
| `Q4_K_S` | 14.5 GB | 4-bit small — good quality/size |
| **`Q4_K_M`** | **15.4 GB** | **recommended — sweet spot** |
| `Q5_K_M` | 17.9 GB | 5-bit — noticeably better than Q4 |
| `Q6_K` | 20.6 GB | 6-bit — near-lossless, tight on 24GB with KV cache |
| `Q8_0` | 26.6 GB | 8-bit — does not fit in 24GB VRAM |

### What "abliterated" means

By default, instruction-tuned models are trained to refuse certain requests
("I can't help with that"). Abliteration removes this refusal behaviour by
identifying the "refusal direction" in the model's weight space — the axis along
which activations shift when the model is about to refuse — and subtracting it
out. It's a post-processing step on the weights, not a fine-tune.

The result is a model that behaves like the base model without a content filter.
For a RAG app answering questions about books, this means no false refusals on
edge-case phrasings.

`huihui-ai` is the most reputable source for abliterated models; Abiray provides
the GGUF quantizations of their work.

### Why it's good for RAG

- **Strong instruction following** — reliably stays within the retrieved context
  and doesn't hallucinate beyond it
- **Long context** — 128K native window; with KV cache quant we use 48K in practice
- **Reasoning** — handles multi-hop questions well (e.g. synthesising across
  several retrieved passages)

```bash
devbox run model-pull
```

## Embedding model: nomic-embed-text-v1.5 (GGUF)

| Property | Value |
|---|---|
| Size | ~370MB (Q8_0 GGUF) |
| Dimensions | 768 |
| Context window | 8192 tokens |
| Benchmark | Top-tier on MTEB retrieval (open-source) |

llama.cpp serves it on port 8081 with `--embedding --pooling mean`. The pooling
mode matters: `mean` averaging over token embeddings gives the best retrieval
quality for this model (matching how it was trained).

**Important:** changing embedding models changes the vector space — all existing
vectors become incompatible. You must re-ingest all documents after any swap.
(snowflake-arctic-embed2 was 1024 dims; nomic-embed-text is 768 dims.)

## Why two llama-server processes?

llama.cpp loads one model per server process. Embedding and generation are separate
models, so they run as `llama-chat` (port 8080) and `llama-embed` (port 8081).
Both expose an OpenAI-compatible API, so the Python code uses `langchain-openai`
with a custom `base_url` — no special Ollama client needed.

## Why llama.cpp over Ollama?

- **Direct control** — flags like `--ctx-size`, `--pooling`, `-ctk` are explicit;
  Ollama hides them behind a modelfile abstraction
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
