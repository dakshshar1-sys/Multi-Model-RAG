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


# ── abstention ───────────────────────────────────────────────────────────────
# Did the answer decline, rather than assert? Used by the abstention evaluation and by the
# knowledge-base answer gate's tests. Heuristic by design: it looks only at the opening of
# the answer, because an answer that opens with a claim and hedges later has still asserted.
_ABSTAIN_RE = re.compile(
    r"\b(?:"
    r"(?:does|do|did)(?:\s+not|n['’]t)\s+(?:contain|mention|provide|specify|include|state|discuss|address|cover|give|list|describe|name|say|appear|indicate|reference)"
    r"|(?:is|are|was|were)(?:\s+not|n['’]t)\s+(?:mentioned|provided|specified|stated|available|included|found|discussed|covered|given|listed|described|named|present|addressed|indicated|referenced)"
    r"|not\s+(?:explicitly\s+)?(?:mentioned|provided|specified|stated|available|included|found|discussed|covered|given|listed|described|named|addressed|indicated)"
    r"|no\s+(?:\w+\s+){0,2}(?:information|mention|details?|data|reference|indication|record|evidence|relevant context)"
    r"|(?:cannot|can['’]?t|could\s+not|couldn['’]t|unable\s+to)\s+(?:find|answer|determine|provide|locate|confirm|identify)"
    r"|i\s+(?:do\s+not|don['’]t)\s+know"
    r"|(?:no\s+one|nobody|none\s+of\s+the\s+(?:sources?|documents?|context))\s+(?:is|are|was|were|has|have|\w+s)\b"
    r"|(?:documents?|context|sources?|knowledge base)\s+(?:do(?:es)?\s+not|don['’]t|doesn['’]t|lacks?)"
    r"|outside\s+the\s+(?:scope|provided)"
    r")", re.I)
ABSTENTION_WINDOW = 320


def is_abstention(answer: str, window: int = ABSTENTION_WINDOW) -> bool:
    """True when the opening of `answer` says the information is not available."""
    return bool(_ABSTAIN_RE.search((answer or "")[:window]))
