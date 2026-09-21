"""
Pure metric functions for the evaluation harness. No models, no I/O, so they are
unit-testable and the numbers in the results chapter are reproducible.

Retrieval:   recall_at_k, mrr          — did the gold evidence come back, and how high?
Answers:     token_f1, contains_gold   — how close is the generated answer to the gold one?
Groundedness: lexical_support          — what fraction of the answer's sentences are
                                          backed by the retrieved context (a model-free
                                          faithfulness proxy, reported next to the
                                          LLM-judge verdict so the two can be compared).
"""
from __future__ import annotations

import re
import string
from collections import Counter

from core.text_support import content_tokens, lexical_support, normalize, token_f1  # noqa: F401  (shared with production)


# ── retrieval ──────────────────────────────────────────────────────────────

def evidence_rank(evidence: str, retrieved_texts: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing the gold evidence
    (whitespace/case-insensitive), or None if absent."""
    ev = normalize(evidence)
    if not ev:
        return None
    for i, chunk in enumerate(retrieved_texts, start=1):
        if ev in normalize(chunk):
            return i
    return None


def recall_at_k(evidence: str, retrieved_texts: list[str], k: int) -> float:
    r = evidence_rank(evidence, retrieved_texts[:k])
    return 1.0 if r is not None else 0.0


def mrr(evidence: str, retrieved_texts: list[str]) -> float:
    r = evidence_rank(evidence, retrieved_texts)
    return 0.0 if r is None else 1.0 / r


# ── answer correctness ─────────────────────────────────────────────────────

def contains_gold(prediction: str, gold: str, min_ratio: float = 0.8) -> float:
    """1.0 if at least `min_ratio` of the gold answer's content tokens appear in the
    prediction. Tolerant of paraphrase, strict on the facts."""
    g = content_tokens(gold)
    if not g:
        return 0.0
    p = set(content_tokens(prediction))
    return 1.0 if sum(t in p for t in g) / len(g) >= min_ratio else 0.0


# ── groundedness ───────────────────────────────────────────────────────────

# ── aggregation ────────────────────────────────────────────────────────────

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
