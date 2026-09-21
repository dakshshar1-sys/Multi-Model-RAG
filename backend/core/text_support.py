"""
Model-free text measures shared by verification and the evaluation harness.

lexical_support(answer, context): the fraction of the answer's sentences whose
content words are mostly present in the context. A sentence made of words that
never appear in the sources is, by construction, not grounded in them. It is
instant, deterministic, and — measured on the evaluation set — decisive at the
extremes: high support means grounded, near-zero means invented. The LLM judge
is only worth its seconds in the band between.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_STOP = set("""a an the and or of to in on for with by from as at is are was were be been being
this that these those it its into than then so if not no nor but which who whom whose what when
where why how do does did done can could should would may might will shall have has had having
we you they he she i me my our your their them his her""".split())
_NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*%?")


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in normalize(text).split() if t not in _STOP and len(t) > 1]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [s.strip() for s in parts if len(content_tokens(s)) >= 3]


def lexical_support(answer: str, context: list[str], min_coverage: float = 0.6) -> float:
    sents = sentences(answer)
    if not sents:
        return 0.0
    ctx = set(content_tokens(" ".join(context)))
    supported = 0
    for s in sents:
        toks = content_tokens(s)
        if sum(t in ctx for t in toks) / len(toks) >= min_coverage:
            supported += 1
    return supported / len(sents)


def count_numbers(text: str) -> int:
    """How many numeric tokens a text contains (what a chart could be drawn from)."""
    return len(_NUM_RE.findall(text or ""))


def token_f1(prediction: str, gold: str) -> float:
    p, g = content_tokens(prediction), content_tokens(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)
