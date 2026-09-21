"""
A small, dependency-free BM25 index over the chunks already held by the vector
store, so retrieval can be hybrid (lexical + dense) without another package.

Why: on the evaluation set, dense-only retrieval put five gold chunks at ranks
12-36 — all short bulleted lines whose embedding is dominated by neighbouring
text — while plain BM25 ranked four of them first or second. Dense top-5 hit 50
of 64 questions, BM25 top-5 hit 60, and the union hit 62.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the and or of to in on for with by from as at is are was were be been being
this that these those it its into than then so if not no nor but which who whom whose what when
where why how do does did done can could should would may might will shall have has had having
we you they he she i me my our your their them his her""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]                      # crude plural strip: matches 'vectors' to 'vector'
        out.append(t)
    return out


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lens: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def build(self, docs: list[tuple[str, str]]) -> "BM25Index":
        """docs: (doc_id, text)."""
        self.ids, self.tfs, self.lens, self.df = [], [], [], Counter()
        for doc_id, text in docs:
            toks = tokenize(text)
            self.ids.append(doc_id)
            self.tfs.append(Counter(toks))
            self.lens.append(len(toks))
            self.df.update(set(toks))
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        return self

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, score) with score > 0, best first."""
        q = tokenize(query)
        if not q or not self.ids:
            return []
        n = len(self.ids)
        idf = {w: math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5)) for w in set(q) if w in self.df}
        if not idf:
            return []
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl) if self.avgdl else self.k1
            for w, w_idf in idf.items():
                f = tf.get(w)
                if f:
                    s += w_idf * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(self.ids[i], s) for s, i in scored[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked id lists: score(id) = sum over lists of 1/(k + rank). Stable on ties."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order; order += 1
    return sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
