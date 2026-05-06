# RAG Evaluation

## Why evaluate?

Running a RAG pipeline feels like it's working — answers come out, they look plausible.
But "looks plausible" is not the same as "correct and grounded". Evaluation makes
quality measurable so you can compare changes (different chunk sizes, retrieval k,
embedding models, prompts) and know whether they actually improved things.

```bash
devbox run eval
```

## What we measure

### Faithfulness
> Does the answer contain only claims supported by the retrieved context?

A RAG system should answer *from* the context, not from the LLM's training data.
Faithfulness catches hallucination — the model confidently stating something that
isn't in the retrieved chunks.

- **1.0** — every claim in the answer is grounded in the context
- **0.0** — the answer ignores or contradicts the context

### Answer relevance
> Does the answer address the question?

Catches refusals ("I can only answer about Sherlock Holmes") and off-topic rambling.
A correct but irrelevant answer scores low.

### Context relevance
> Do the retrieved chunks contain the information needed to answer the question?

This measures the **retriever**, not the LLM. If context relevance is low but
faithfulness is high, the LLM is hallucinating to fill the gap. If context relevance
is high but faithfulness is low, the LLM is ignoring what it retrieved.

### Correctness
> Does the answer match a known reference answer?

Only applies to questions with a known ground-truth reference. The other three
metrics don't require reference answers — they're *reference-free*, which is why
they scale: you don't need to hand-label thousands of Q&A pairs.

## How LLM-as-judge works

Instead of hardcoded rules, we ask the LLM itself to score each metric. The judge
prompt describes the scoring criteria and asks for a JSON response:

```
{"score": 0.85, "reason": "The answer mentions cocaine but doesn't cite the story."}
```

The LLM evaluating its own output sounds circular, but it works in practice because:
- The judge sees the *context* and *reference* alongside the answer
- Scoring is easier than generating — the judge just needs to compare, not create
- For factual corpora like fiction, the scores correlate well with human judgement

## Limitations of LLM-as-judge

- **Bias toward fluency** — the judge may score a well-written wrong answer higher
  than a correct but clunky one
- **Same model bias** — if the judge and the generator are the same model, the judge
  may be lenient on its own outputs. Ideally use a different (stronger) model as judge.
- **Score calibration varies** — 0.7 from one judge prompt is not directly comparable
  to 0.7 from another. Use scores relatively (before vs after a change), not absolutely.

## Why not RAGAS?

RAGAS is a popular evaluation framework for RAG that implements these same metrics.
We implement them from scratch here because:
1. It's a learning exercise — seeing the prompts directly teaches what the metrics mean
2. RAGAS uses the OpenAI API by default; wiring it to a local llama.cpp server adds
   configuration overhead that obscures the concept
3. Our custom implementation is ~100 lines and easy to modify

Once you understand the concepts, RAGAS is worth exploring for production use.

## Interpreting results

Run eval after each significant change and compare averages:

| Change | What to watch |
|---|---|
| Smaller/larger chunks | context_relevance, faithfulness |
| Different embedding model | context_relevance |
| Different retrieval k | context_relevance, faithfulness |
| Different LLM | faithfulness, answer_relevance, correctness |
| Prompt changes | faithfulness, answer_relevance |

A common failure pattern: **high answer_relevance, low faithfulness** — the model
gives a confident, relevant-sounding answer but is drawing on training data rather
than the retrieved context. Fix: tighten the answer prompt to say "only use the
context below" and evaluate faithfulness again.
