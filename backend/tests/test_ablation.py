"""Ablation runner: the ladder, scoring and the table (no models)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.ablation import LADDER, _aggregate, _score, format_table


def test_ladder_adds_one_component_at_a_time():
    names = [n for n, *_ in LADDER]
    assert names == ["no_retrieval", "dense", "dense+rerank", "hybrid", "hybrid+rerank"]
    assert LADDER[0][1] is None and LADDER[-1][1] == "hybrid" and LADDER[-1][3] is True


def test_score_and_aggregate_and_table():
    q = {"id": "q1", "gold_answer": "FAISS is the vector database", "gold_evidence": "vector database"}
    ctx = ["FAISS is the vector database used by the system."]
    good = _score(q, "The vector database is FAISS.", ctx, 1.5)
    bad = _score(q, "It uses Pinecone.", [], 0.5)
    assert good["contains_gold"] == 1.0 and good["evidence_in_context"] is True and good["lexical_support"] == 1.0
    assert bad["contains_gold"] == 0.0 and bad["evidence_in_context"] is False and bad["lexical_support"] == 0.0
    agg = _aggregate("dense", [good, bad])
    assert agg["contains_gold"] == 0.5 and agg["n"] == 2 and agg["seconds"] == 1.0
    table = format_table([agg, {**agg, "row": "full_e2e", "tools": {"Search_Knowledge_Base": 2}}])
    assert "| dense | 0.500 |" in table and "full_e2e (tools: {'Search_Knowledge_Base': 2})" in table


def test_heading_sheet_extracts_titles_not_body_text(tmp_path):
    from eval.make_heading_sheet import headings
    doc = tmp_path / "d.txt"
    doc.write_text("Chapter 1: EXECUTIVE SUMMARY\n\nThe system is described in prose here, at length, with details.\n"
                   "2.3 Vector Database: FAISS\nFAISS-CPU is utilised for its speed.\nPage | 3\nDept of CSE, JIT, Bengaluru   2026-27   Page 1\n"
                   "LITERATURE SURVEY\n3.1 Functional requirements:\n")
    hs = headings(str(doc))
    assert "Chapter 1: EXECUTIVE SUMMARY" in hs and "2.3 Vector Database: FAISS" in hs and "LITERATURE SURVEY" in hs
    assert "3.1 Functional requirements:" in hs
    assert not any("prose" in h or "utilised" in h or "Page |" in h or "Dept of CSE" in h for h in hs)
