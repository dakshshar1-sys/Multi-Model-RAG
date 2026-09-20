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

_STOP = set("""a an the and or of to in on for with by from as at is are was were be been being
this that these those it its into than then so if not no nor but which who whom whose what when
where why how do does did done can could should would may might will shall have has had having
we you they he she i me my our your their them his her""".split())


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def _stem(tok: str) -> str:
    """Conservative plural/inflection strip so 'vectors' matches 'vector'. Not a real
    stemmer on purpose: it must never merge distinct terms."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in normalize(text).split() if t not in _STOP and len(t) > 1]


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

def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token F1 on content tokens."""
    p, g = content_tokens(prediction), content_tokens(gold)
    if not p or not g:
        return 0.0   # an empty prediction earns nothing; an empty gold is unscorable
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str, min_ratio: float = 0.8) -> float:
    """1.0 if at least `min_ratio` of the gold answer's content tokens appear in the
    prediction. Tolerant of paraphrase, strict on the facts."""
    g = content_tokens(gold)
    if not g:
        return 0.0
    p = set(content_tokens(prediction))
    return 1.0 if sum(t in p for t in g) / len(g) >= min_ratio else 0.0


# ── groundedness ───────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [s.strip() for s in parts if len(content_tokens(s)) >= 3]


def lexical_support(answer: str, context: list[str], min_coverage: float = 0.6) -> float:
    """
    Fraction of the answer's sentences whose content tokens are at least
    `min_coverage` covered by the context. A sentence made of words that never
    appear in the sources is, by construction, not grounded in them. Cheap,
    deterministic, and reported alongside the LLM judge — where the two disagree
    is exactly what to read by hand.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    ctx = set(content_tokens(" ".join(context)))
    supported = 0
    for s in sents:
        toks = content_tokens(s)
        if sum(t in ctx for t in toks) / len(toks) >= min_coverage:
            supported += 1
    return supported / len(sents)


# ── aggregation ────────────────────────────────────────────────────────────

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
