"""Evaluate RAG quality using LLM-as-judge on a fixed question set.

Metrics scored 0.0–1.0 by the LLM:
  faithfulness      — answer contains only claims supported by the retrieved context
  answer_relevance  — answer actually addresses the question
  context_relevance — retrieved chunks contain information needed to answer
  correctness       — answer matches a known reference answer (where provided)
"""

import json
import os
import uuid

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph import build_graph, stream_answer

LLM_HOST = os.environ.get("LLM_HOST", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-27b-instruct")

QUESTIONS = [
    {"question": "Who is Sherlock Holmes's landlady?",
     "reference": "Mrs. Hudson"},
    {"question": "What is the address of Sherlock Holmes?",
     "reference": "221B Baker Street, London"},
    {"question": "Who is Dr. Watson?",
     "reference": "Dr. John H. Watson is Holmes's friend, biographer, and former army surgeon."},
    {"question": "What is the name of Holmes's brother?",
     "reference": "Mycroft Holmes"},
    {"question": "What drug did Sherlock Holmes use?",
     "reference": "Cocaine (a 7% solution) and sometimes morphine."},
    {"question": "In which story does Holmes fake his own death?",
     "reference": "The Final Problem — Holmes falls at the Reichenbach Falls fighting Moriarty."},
    {"question": "Who is Professor Moriarty?",
     "reference": "The criminal mastermind and arch-nemesis of Sherlock Holmes."},
    {"question": "What is the Hound of the Baskervilles?",
     "reference": "A giant hound said to haunt the Baskerville family on the Dartmoor moors."},
]

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate whether the ANSWER contains only claims supported by the CONTEXT. "
    "A score of 1.0 means every claim in the answer is grounded in the context. "
    "A score of 0.0 means the answer contradicts or ignores the context.\n\n"
    "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n\n"
    'Reply with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
)

_ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate whether the ANSWER addresses the QUESTION. "
    "A score of 1.0 means the answer is on-topic and complete. "
    "A score of 0.0 means the answer is irrelevant or a refusal.\n\n"
    "QUESTION: {question}\n\nANSWER: {answer}\n\n"
    'Reply with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
)

_CONTEXT_RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "Evaluate whether the CONTEXT contains the information needed to answer the QUESTION. "
    "A score of 1.0 means the context is highly relevant. "
    "A score of 0.0 means the context is unrelated.\n\n"
    "QUESTION: {question}\n\nCONTEXT:\n{context}\n\n"
    'Reply with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
)

_CORRECTNESS_PROMPT = ChatPromptTemplate.from_template(
    "Compare the ACTUAL ANSWER to the REFERENCE ANSWER for the given QUESTION. "
    "A score of 1.0 means the actual answer is factually equivalent to the reference. "
    "A score of 0.0 means it is factually wrong or a refusal.\n\n"
    "QUESTION: {question}\n\nREFERENCE: {reference}\n\nACTUAL ANSWER: {answer}\n\n"
    'Reply with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}'
)


def _judge(prompt: ChatPromptTemplate, **kwargs) -> tuple[float, str]:
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_HOST, api_key="sk-local")
    try:
        result = (prompt | llm | JsonOutputParser()).invoke(kwargs)
        return float(result["score"]), result.get("reason", "")
    except Exception as exc:
        return 0.0, f"judge error: {exc}"


def _run_one(graph, q: dict) -> dict:
    answer = ""
    documents = []
    for event, payload in stream_answer(q["question"], graph, str(uuid.uuid4())):
        if event == "docs":
            documents = payload
        elif event == "token":
            answer += payload

    context = "\n\n".join(doc.page_content for doc in documents)

    f_score, f_reason = _judge(_FAITHFULNESS_PROMPT, question=q["question"], context=context, answer=answer)
    ar_score, ar_reason = _judge(_ANSWER_RELEVANCE_PROMPT, question=q["question"], answer=answer)
    cr_score, cr_reason = _judge(_CONTEXT_RELEVANCE_PROMPT, question=q["question"], context=context)

    result = {
        "question": q["question"],
        "answer": answer,
        "context_chunks": len(documents),
        "faithfulness": f_score,
        "answer_relevance": ar_score,
        "context_relevance": cr_score,
        "faithfulness_reason": f_reason,
        "answer_relevance_reason": ar_reason,
        "context_relevance_reason": cr_reason,
    }

    if "reference" in q:
        c_score, c_reason = _judge(_CORRECTNESS_PROMPT, question=q["question"], reference=q["reference"], answer=answer)
        result["correctness"] = c_score
        result["correctness_reason"] = c_reason

    return result


def run_eval() -> list[dict]:
    graph = build_graph()
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [{i}/{len(QUESTIONS)}] {q['question'][:70]}", end="\r", flush=True)
        results.append(_run_one(graph, q))
    print()
    return results


def _print_report(results: list[dict]) -> None:
    metrics = ["faithfulness", "answer_relevance", "context_relevance", "correctness"]
    active = [m for m in metrics if any(m in r for r in results)]
    col = 18

    header = f"{'Question':<42} " + "  ".join(f"{m[:col]:>{col}}" for m in active)
    sep = "-" * len(header)
    print(header)
    print(sep)

    totals: dict[str, list[float]] = {m: [] for m in active}
    for r in results:
        row = f"{r['question'][:41]:<42}"
        for m in active:
            if m in r:
                row += f"  {r[m]:>{col}.2f}"
                totals[m].append(r[m])
            else:
                row += f"  {'N/A':>{col}}"
        print(row)

    print(sep)
    avg_row = f"{'AVERAGE':<42}"
    for m in active:
        if totals[m]:
            avg_row += f"  {sum(totals[m]) / len(totals[m]):>{col}.2f}"
    print(avg_row)

    print("\n=== Reasoning ===\n")
    for r in results:
        print(f"Q: {r['question']}")
        print(f"   faithfulness:      {r['faithfulness']:.2f}  {r['faithfulness_reason']}")
        print(f"   answer_relevance:  {r['answer_relevance']:.2f}  {r['answer_relevance_reason']}")
        print(f"   context_relevance: {r['context_relevance']:.2f}  {r['context_relevance_reason']}")
        if "correctness" in r:
            print(f"   correctness:       {r['correctness']:.2f}  {r['correctness_reason']}")
        print()


if __name__ == "__main__":
    print(f"Running {len(QUESTIONS)} evaluation questions...\n")
    results = run_eval()
    print("\n=== Scores ===\n")
    _print_report(results)
