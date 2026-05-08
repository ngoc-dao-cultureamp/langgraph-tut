# Models

## Choosing a model for local RAG

Key criteria when selecting a local LLM for RAG:

- **Memory fit** — model weights + KV cache must fit in VRAM (GPU) or unified RAM (Apple Silicon). Rule of thumb: Q4 quantization ≈ 0.5 GB per billion parameters.
- **Instruction following** — the model must stay within retrieved context and not hallucinate beyond it.
- **Context window** — RAG prompts include retrieved chunks; you need headroom above the chunk content.
- **Speed** — for interactive use, generation speed matters as much as quality.

MoE (Mixture-of-Experts) models activate only a fraction of their weights per token, making them faster than dense models of equivalent total size on memory-bandwidth-limited hardware (Apple Silicon, consumer GPUs).

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

Instruction-tuned models are trained to refuse certain requests ("I can't help with that"). For a RAG app this matters because edge-case phrasings of legitimate questions can trigger false refusals.

#### Community naming conventions

- **"Abliterated"** — weights were post-processed using an activation-space technique (see below). No new training data involved.
- **"Uncensored"** — usually means fine-tuned on a dataset that excluded refusal examples (e.g. Eric Hartford's uncensored datasets). Sometimes used loosely to mean abliterated.
- **"Heretic"** — a specific community variant (Youssofal) that combines abliteration with additional merging steps.
- These terms are **not standardised** — always check the model card.

#### Post-processing techniques (no retraining)

Both work by finding the "refusal direction" in the model's residual stream and projecting it out of the weight matrices. The difference is *how* that direction is found:

| Technique | How direction is found | Notes |
|---|---|---|
| **Mean-diff abliteration** (Failspy, 2024) | Average activations on "harmful" prompts minus average on "harmless" prompts | Classic, fast, well-understood |
| **Wasserstein abliteration** (LuffyTheFox) | Optimal-transport geometry between harmful/harmless activation distributions | Newer; claims more thorough removal with less capability damage |

Neither is a fine-tune — they directly modify weight tensors. The model tokenizer/architecture is unchanged; only floating-point values shift.

#### Fine-tune based uncensoring

Train (or LoRA-adapt) the model on curated datasets that demonstrate helpful responses to refused topics. Slower to produce, but can yield more nuanced behaviour than pure activation surgery. Eric Hartford's `WizardLM-uncensored` datasets pioneered this approach.

#### Merge-based approaches

Blend a safety-trained model with a base (pre-RLHF) checkpoint using techniques like SLERP, TIES, or DARE. The safety layers get diluted by the base weights. Quality depends heavily on the merge ratio and which layers are targeted.

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

We use `-ctk q8_0 -ctv q8_0`. The `_0` suffix means basic linear quantization — the only variant used for KV cache because activations are transient and fancier schemes aren't worth the overhead.

### KV cache budget calculation

To estimate usable context for a given model:

```
KV cache per token = 2 × KV_heads × head_dim × layers × bytes_per_element
```

Example for a 27B model with 4 KV heads, 256 head dim, 64 layers, fp16:
```
2 × 4 × 256 × 64 × 2 bytes = 256 KB per token
```

With 8.6 GB headroom (24 GB VRAM − 15.4 GB weights):

| KV cache type | Usable context |
|---|---|
| fp16 | ~34K tokens |
| q8_0 | ~68K tokens |
| q4_0 | ~136K tokens |

This calculation applies to any transformer model — just substitute the right architectural constants from the model card.

## Why two inference server processes?

Embedding and generation are separate models, so they run as separate server processes (e.g. on different ports). Both expose an OpenAI-compatible API, so the Python code uses a single client with different `base_url` values — no special client per model needed.

## AWS Bedrock (cloud alternative)

Replace `ChatOpenAI` with `ChatBedrock` when deploying to AWS. Good choices:
- High quality: `anthropic.claude-3-5-sonnet` family
- Cost-effective: `amazon.nova-pro-v1`

For embeddings, swap `OpenAIEmbeddings` for `BedrockEmbeddings` (e.g. `amazon.titan-embed-text-v2`). Note: dimensions differ between embedding models, so re-ingest all documents when switching.
