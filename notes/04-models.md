# Models

## Hardware

| Machine | Chip | Memory | Backend |
|---|---|---|---|
| MacBook | Apple M3 Max | 36GB unified RAM | Metal (via llama.cpp `-ngl 99`) |
| Workstation | NVIDIA RTX 3090 | 24GB VRAM | CUDA (via llama.cpp `-ngl 99`) |

## Concepts

### GGUF quantization

GGUF is the file format llama.cpp uses. Quantization reduces precision to save memory.

**Name anatomy: `Q4_K_M`**
- `Q4` — 4 bits per weight element (vs 16 bits for fp16)
- `K` — "K-quant": groups weights with shared scale factors, preserving more information than naive rounding
- `M` — medium variant (S = small, M = medium, L = large within the same bit level)

**Variants:**
- **`IQ` (importance-weighted):** `IQ4_XS`, `IQ4_NL` — assign more bits to weights that matter most. Better quality per GB, slightly slower to load.
- **`UD` (Unsloth Dynamic):** `UD-Q4_K_XL` — layers earlier/later in the network get more bits; middle layers fewer. Best quality for a given file size.
- **`_K_P` (K-quant Plus):** Critical layers (attention, embeddings) kept at higher precision. Similar to UD but applied uniformly by layer type rather than by measured importance.

### Refusal removal

Instruction-tuned models are trained to refuse certain requests. Two main techniques:

- **Mean-diff abliteration** — find the refusal direction via difference of means in activation space, subtract it from the weights. Classic, well-understood.
- **Wasserstein distance** — uses optimal-transport geometry to find the refusal direction. Newer; claims more thorough removal with less collateral damage to capability.

Both are post-processing steps on the weights, not fine-tunes. For a RAG app, this prevents false refusals on edge-case question phrasings.

### Model weight formats

| Format | Description | Use case |
|---|---|---|
| **Safetensors** | Standard HF format, full precision (bf16/fp32). Safe from pickle exploits. | Fine-tuning, conversion source |
| **MLX** | Apple M-series optimised format. Runs via `mlx-lm`. Not portable to other hardware. | Apple Silicon inference |
| **GGUF** | Single-file bundle (weights + tokenizer + metadata). Quantization-friendly. | llama.cpp inference ← what we use |

GGUF requires conversion from safetensors (`llama.cpp`'s `convert_hf_to_gguf.py` + quantize). New model releases often have safetensors/MLX first; GGUFs appear within days as community members run the conversion.

### KV cache

The KV cache stores intermediate attention state during generation. It grows with context length — longer conversations or more retrieved chunks = more memory needed.

llama.cpp can quantize the KV cache to extend usable context:

| `-ctk`/`-ctv` type | Bytes/element | Notes |
|---|---|---|
| fp16 (default) | 2 | Full precision |
| q8_0 | 1 | Minimal quality loss |
| q4_0 | 0.5 | Slight quality loss |

We use `-ctk q8_0 -ctv q8_0` (set in `process-compose.yaml`). The `_0` suffix means basic linear quantization — the only variant used for KV cache because activations are transient and fancier schemes aren't worth the overhead.

## Current LLM: Qwen3.6-35B-A3B (MoE, uncensored)

Qwen3.6 is a model family from Alibaba. "3.6" is a generation name, not a parameter count. The 35B-A3B is the **MoE** (Mixture-of-Experts) model in the family.

"A3B" = ~3B parameters **active per token** out of 35B total weights. On Apple Silicon (memory-bandwidth-limited), fewer active params per token means faster generation than a comparably-sized dense model.

### Specs

| Property | Value |
|---|---|
| Parameters (total / active) | 35B / ~3B per token |
| Architecture | Mixture-of-Experts (MoE) |
| Context window | 128K (we use 48K with KV cache quant) |
| Disk size (Q4_K_P GGUF) | ~23.4 GB |
| License | Apache 2.0 |

### Variant: LuffyTheFox Wasserstein Q4_K_P

`LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Wasserstein-GGUF` — chosen over `Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-GGUF` (Q4_K_M) because:
- Flat file layout — simpler `hf download`
- `_K_P` quant gives a slight quality edge over `_K_M`
- More community adoption (237k vs 25k downloads)

| File | Size |
|---|---|
| `Qwen3.6-35B-A3B-Uncensored.IQ3_M.gguf` | 15.4 GB |
| `Qwen3.6-35B-A3B-Uncensored.Q3_K_P.gguf` | 19 GB |
| `Qwen3.6-35B-A3B-Uncensored.IQ4_NL.gguf` | 19.8 GB |
| `Qwen3.6-35B-A3B-Uncensored.MXFP4_MOE.gguf` | 21.7 GB |
| **`Qwen3.6-35B-A3B-Uncensored.Q4_K_P.gguf`** | **23.4 GB** ← in use |
| `Qwen3.6-35B-A3B-Uncensored.Q5_K_P.gguf` | 28 GB |
| `Qwen3.6-35B-A3B-Uncensored.Q6_K_P.gguf` | 30.6 GB |
| `Qwen3.6-35B-A3B-Uncensored.Q8_K_P.gguf` | 43.6 GB |

### Why it's good for RAG

- **MoE speed** — fewer active params per token = faster on Apple Silicon
- **Strong instruction following** — stays within retrieved context, doesn't hallucinate beyond it
- **Long context** — 128K native; we use 48K in practice with KV cache quantization
- **Reasoning** — handles multi-hop questions well across several retrieved passages

```bash
devbox run model-pull
```

## Previous LLM: Qwen3.6-27B (dense, abliterated)

`Abiray/Huihui-Qwen3.6-27B-abliterated-GGUF` — Q4_K_M (~15.4 GB). Switched to 35B-A3B for better speed on Apple Silicon.

Dense (non-MoE) model — all 27B parameters are used every token.

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

### KV cache calculation

```
KV cache per token = 2 × KV_heads × head_dim × layers × bytes_per_element
                   = 2 × 4 × 256 × 64 × 2 bytes (fp16)
                   = 256 KB per token
```

With 15.4 GB for weights and 24 GB VRAM total, ~8.6 GB headroom gives:

| KV cache type | Usable context |
|---|---|
| fp16 | ~34K tokens |
| q8_0 | ~68K tokens |
| q4_0 | ~136K tokens |

Using q8_0, the target was `--ctx-size 49152` (48K) — safe margin below 68K theoretical max. 48K ≈ one complete Sherlock Holmes novel.

### Available quantizations

| File | Size | Notes |
|---|---|---|
| `Q3_K_M` | 12.4 GB | 3-bit — noticeable quality drop |
| `Q4_K_S` | 14.5 GB | 4-bit small |
| **`Q4_K_M`** | **15.4 GB** | **was in use** |
| `Q5_K_M` | 17.9 GB | 5-bit — noticeably better than Q4 |
| `Q6_K` | 20.6 GB | 6-bit — near-lossless |
| `Q8_0` | 26.6 GB | Does not fit in 24 GB VRAM |

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
