"""
Hybrid retrieval: dense + BM25 with reciprocal rank fusion.

Background: on the evaluation set, dense-only retrieval ranked five gold chunks at
12-36 (short bulleted lines) while BM25 put four of them at 1-2; dense top-5 hit 50/64,
BM25 60/64, the union 62/64.

Run:  cd backend && python -m pytest tests/test_hybrid_retrieval.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25 import BM25Index, reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_drops_stopwords_and_strips_plurals():
    assert tokenize("The Vectors are stored in FAISS, 384 dimensions!") == ["vector", "stored", "faiss", "384", "dimension"]
    assert tokenize("") == []
    assert "class" in tokenize("class") and "clas" not in tokenize("class"), "double-s words keep their s"


def test_bm25_ranks_the_exact_term_chunk_first():
    idx = BM25Index().build([
        ("a", "Retrieval-level errors: irrelevance, overlap, omission."),
        ("b", "Generation-level errors: hallucination, creating facts; drift, losing query focus."),
        ("c", "The frontend uses Tailwind CSS and Lucide icons for the timeline."),
    ])
    top = idx.search("which errors involve drift and losing query focus?", top_k=3)
    assert top[0][0] == "b"
    assert all(s > 0 for _, s in top)
    assert idx.search("zzz qqq", top_k=3) == [], "no overlap: no results, not noise"
    assert idx.search("", top_k=3) == []


def test_rrf_merges_and_orders_by_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] in ("a", "c") and set(fused[:2]) == {"a", "c"}, "items in both lists rise"
    assert fused[-1] in ("b", "d")
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([["x"], []]) == ["x"]


@pytest.fixture
def db(tmp_path):
    from langchain_core.documents import Document
    from retrieval.vector_db import VectorDatabase
    d = VectorDatabase(index_path=str(tmp_path / "idx"))
    d.add_documents([
        Document(page_content="A long paragraph about enterprise knowledge management, document silos, and how retrieval augmented generation grounds answers in sources.", metadata={"source": "a"}),
        Document(page_content="- Irrelevance: retrieving the wrong document (e.g. Finance as Legal)\n- Overlap: redundant chunks\n- Omission: missing context", metadata={"source": "b"}),
        Document(page_content="The verifier flags a FAIL if it sees $40B instead of $100B in the answer.", metadata={"source": "c"}),
        Document(page_content="Tailwind CSS provides utility-first styling with dark mode and glassmorphism.", metadata={"source": "d"}),
    ])
    return d


def test_hybrid_finds_the_exact_term_chunk_and_returns_documents(db):
    docs = db.hybrid_retrieve("what does the verifier flag if it sees $40B instead of $100B?", top_k=2)
    assert docs and docs[0].metadata["source"] == "c"
    lex = db.bm25_retrieve("Finance as Legal irrelevance example", top_k=1)
    assert lex[0].metadata["source"] == "b"


def test_hybrid_never_returns_duplicates_and_respects_top_k(db):
    docs = db.hybrid_retrieve("retrieval errors and enterprise documents", top_k=3)
    texts = [d.page_content for d in docs]
    assert len(texts) == len(set(texts)) and len(docs) <= 3


def test_lexical_index_tracks_adds_and_deletes(db):
    from langchain_core.documents import Document
    assert len(db._ensure_bm25()) == 4
    db.add_documents([Document(page_content="Framer Motion animates the timeline component transitions.", metadata={"source": "e"})])
    assert len(db._ensure_bm25()) == 5
    assert db.bm25_retrieve("Framer Motion timeline", top_k=1)[0].metadata["source"] == "e"
    assert db.delete_by_source("e") == 1
    assert len(db._ensure_bm25()) == 4
    assert all(d.metadata["source"] != "e" for d in db.bm25_retrieve("Framer Motion timeline", top_k=5))
