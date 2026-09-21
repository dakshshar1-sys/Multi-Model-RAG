"""Unit tests for the evaluation harness metrics (pure functions, no models)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (contains_gold, evidence_rank, lexical_support, mean, mrr,
                          recall_at_k, token_f1)

CHUNKS = ["Intro text about RAG systems.",
          "FAISS (Facebook AI Similarity Search) is the vector database used.",
          "Ollama hosts the local LLM."]


def test_evidence_rank_is_case_and_whitespace_insensitive():
    assert evidence_rank("facebook ai   similarity search", CHUNKS) == 2
    assert evidence_rank("Ollama hosts", CHUNKS) == 3
    assert evidence_rank("Pinecone", CHUNKS) is None
    assert evidence_rank("", CHUNKS) is None


def test_recall_and_mrr():
    assert recall_at_k("FAISS", CHUNKS, k=1) == 0.0
    assert recall_at_k("FAISS", CHUNKS, k=2) == 1.0
    assert mrr("FAISS", CHUNKS) == 0.5
    assert mrr("nothing here", CHUNKS) == 0.0


def test_token_f1_and_containment():
    assert token_f1("FAISS is the vector database", "the vector database is FAISS") == 1.0
    assert 0.0 < token_f1("FAISS is used", "FAISS is the vector database used") < 1.0
    assert token_f1("", "x") == 0.0
    assert contains_gold("They use FAISS as the vector database here", "FAISS vector database") == 1.0
    assert contains_gold("They use Pinecone", "FAISS vector database") == 0.0


def test_lexical_support_flags_unsupported_sentences():
    ctx = ["FAISS is the vector database. Ollama hosts the local model."]
    grounded = "Vectors live in the FAISS vector database. Ollama hosts the local model."
    invented = "Vectors live in the FAISS vector database. It was deployed to Kubernetes on Azure last spring."
    assert lexical_support(grounded, ctx) == 1.0
    assert lexical_support(invented, ctx) == 0.5
    assert lexical_support("", ctx) == 0.0


def test_mean():
    assert mean([1.0, 0.0]) == 0.5 and mean([]) == 0.0
