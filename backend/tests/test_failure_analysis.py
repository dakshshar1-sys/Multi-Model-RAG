"""The failure-analysis appendix is built from result files, never by hand."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.failure_analysis import build

QA = {"q1": {"question": "What is FAISS?", "source": "a.txt", "gold_evidence": "vector database"},
      "q2": {"question": "Who hosts the model?", "source": "a.txt", "gold_evidence": "Ollama hosts"}}
ANSWERS = {"answers": {"rows": [
    {"id": "q1", "gold": "FAISS is the vector database", "answer": "It stores vectors in Pinecone.", "evidence_in_context": True, "contains_gold": 0.0, "lexical_support": 0.2, "judge_pass": True, "judge_reason": "fine"},
    {"id": "q2", "gold": "Ollama", "answer": "Ollama hosts the local model.", "evidence_in_context": True, "contains_gold": 1.0, "lexical_support": 1.0, "judge_pass": True, "judge_reason": "ok"}],
    "proxy_vs_judge_disagreements": ["q1"]}}
RET_D = {"retrieval": {"top_k_final": 5, "rows": [{"id": "q1", "pre_recall@5": 0.0, "pre_rank": None}, {"id": "q2", "pre_recall@5": 1.0, "pre_rank": 1}]}}
RET_H = {"retrieval": {"top_k_final": 5, "rows": [{"id": "q1", "pre_recall@5": 1.0, "pre_rank": 2}, {"id": "q2", "pre_recall@5": 1.0, "pre_rank": 1}]}}
ROUTING = {"routing": {"accuracy_mean": 0.9, "unstable_cases": ["r6"], "confusion": [{"expected": "Send_Email", "got": "Send_Telegram", "count": 2}],
                       "rows": [{"expected": "Send_Email", "got": "Send_Telegram", "query": "email Ali about the meeting"}]}}
ABL = {"per_item": {"no_retrieval": [{"id": "q1", "answer": "FAISS is a library by Meta", "contains_gold": 0.0}, {"id": "q2", "answer": "Ollama", "contains_gold": 1.0}]}}


def test_every_section_is_present_and_rows_are_real_outputs():
    md = build(QA, ANSWERS, RET_D, RET_H, ROUTING, ABL)
    assert "## 1. Retrieval misses (dense top-5) — 1 of 2" in md and "| q1 | What is FAISS? | a.txt | vector database | None | 2 |" in md
    assert "## 2. Generation faults" in md and "Pinecone" in md
    assert "## 3. LLM judge vs lexical proxy disagreements — 1" in md and "| q1 | 0.20 | PASS |" in md
    assert "## 4. Routing confusions" in md and "email Ali about the meeting" in md and "r6" in md
    assert "## 5. Model alone" in md and "1 of 2 contain the gold fact" in md


def test_missing_result_files_are_tolerated():
    md = build(QA, None, None, None, None, None)
    assert md.startswith("# Failure analysis") and "## 1." not in md
