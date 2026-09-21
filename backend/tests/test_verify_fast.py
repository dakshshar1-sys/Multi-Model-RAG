"""Verification tries the model-free proxy first and consults the model only when unsure."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_support import count_numbers, lexical_support
from verification.verifier import VerificationModule

CTX = ["FAISS is the vector database. Ollama hosts the local model. The reranker is a cross-encoder."]


def _module(monkeypatch, llm_reply="Result: PASS\nReason: fine"):
    v = VerificationModule.__new__(VerificationModule)
    calls = []
    class LLM:
        def invoke(self, prompt, model_choice="auto"):
            calls.append(prompt); return llm_reply
    v.llm = LLM()
    from langchain_core.prompts import PromptTemplate
    v.prompt_template = PromptTemplate(input_variables=["context", "answer"], template="{context}\n{answer}")
    return v, calls


def test_grounded_answer_passes_without_a_model_call(monkeypatch):
    v, calls = _module(monkeypatch)
    ok, reason = asyncio.run(v.verify_fast("FAISS is the vector database. Ollama hosts the local model.", CTX))
    assert ok and calls == [] and "lexical support" in reason


def test_invented_answer_fails_without_a_model_call(monkeypatch):
    v, calls = _module(monkeypatch)
    ok, reason = asyncio.run(v.verify_fast("The system was deployed to Kubernetes on Azure last spring by a team of twelve.", CTX))
    assert not ok and calls == []


def test_uncertain_answer_consults_the_model_with_trimmed_context(monkeypatch):
    v, calls = _module(monkeypatch, "Result: FAIL\nReason: numbers invented")
    mixed = "FAISS is the vector database. It was deployed to Kubernetes on Azure last spring."   # 0.5 support
    big_ctx = ["x" * 5000] * 6
    ok, reason = asyncio.run(v.verify_fast(mixed, CTX + big_ctx))
    assert calls, "model consulted in the uncertain band"
    assert len(calls[0]) < 3 * 1500 + 2000, "context trimmed to the first chunks"
    assert not ok and "model consulted" in reason


def test_count_numbers():
    assert count_numbers("Jan 100, Feb 75, Mar 50") == 3
    assert count_numbers("no numbers here") == 0
    assert count_numbers("revenue rose 12.5% to $93.8 trillion in Q4") == 2
