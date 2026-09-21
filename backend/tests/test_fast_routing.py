"""
Deterministic routing for unmistakable intents: each case in the evaluation set that
a rule covers must hit the rule (no model call), and the rules must never hijack a
send into a file write or a read.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agentic_router import AgentRouter

fast = AgentRouter._fast_route
CASES = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "eval", "datasets", "routing.jsonl")) if l.strip()]
COVERED = {"file-create", "file-edit", "send", "send-mentions-file", "send-whatsapp-alias", "inbox", "chart-inline", "kb", "vision", "chat", "ambiguous"}


@pytest.mark.parametrize("case", [c for c in CASES if c["tag"] in COVERED], ids=lambda c: c["id"])
def test_every_covered_eval_case_is_routed_deterministically(case):
    assert fast(case["query"]) == case["expected"], case["query"]


@pytest.mark.parametrize("case", [c for c in CASES if c["tag"] not in COVERED], ids=lambda c: c["id"])
def test_uncovered_cases_fall_through_to_the_model_or_agree(case):
    got = fast(case["query"])
    assert got is None or got == case["expected"] or got in case.get("accept", []), (case["query"], got)


def test_send_beats_read_and_file():
    assert fast("email Ali the config.py we discussed") == "Send_Email"
    assert fast("what did my supervisor email me about?") == "Read_Email"
    assert fast("send Ali a message about main.py") == "Send_Telegram"
    assert fast("email me the summary") is None, "sending to oneself is not unmistakable"


def test_greeting_with_a_task_is_not_chat():
    assert fast("hello, search my uploaded documents for the fee structure") == "Search_Knowledge_Base"
    assert fast("hi, what's the weather in Bengaluru today?") is None


def test_chart_needs_numbers_in_the_message():
    assert fast("make a bar chart: Jan 100, Feb 75") == "Visualize_Data"
    assert fast("chart Samsung's 2025 revenue by quarter") is None, "one number only: needs a lookup, model decides"


# ── retrieval-first routing: the knowledge-base probe ───────────────────────
# Background: the whole-pipeline ablation scored 0.094 end to end because reader-style
# questions ("according to the report, ...") went to web search 57/64 times. Measured:
# corpus questions score median +4.4 on the reranker's top candidate, web-bound ones
# median -11.0; threshold -3 routes 94% of corpus questions to the KB and 0% of web.

from models.agentic_router import AgentRouter, KB_PROBE_THRESHOLD


def _router_with(probe, monkeypatch, llm_answer="Web_Search"):
    r = AgentRouter.__new__(AgentRouter)
    class LLM:
        def invoke(self, prompt, model_choice="auto"): return llm_answer
    r.llm = LLM(); r.prompt_template = type("T", (), {"format": lambda self, **k: "p"})()
    r.kb_probe = probe; r.last_probe_score = None
    return r


def test_strong_kb_match_routes_to_the_knowledge_base_without_the_classifier(monkeypatch):
    r = _router_with(lambda q: 4.4, monkeypatch)
    assert r.route_query("according to the survey report, what were early RAG systems based on?", "auto") == "Search_Knowledge_Base"
    assert r.last_probe_score == 4.4


def test_weak_kb_match_falls_through_to_the_classifier(monkeypatch):
    r = _router_with(lambda q: -11.0, monkeypatch, llm_answer="Web_Search")
    assert r.route_query("what is the capital of Australia?", "auto") == "Web_Search"
    assert r.last_probe_score == -11.0


def test_live_intent_bypasses_the_probe_even_on_a_strong_match(monkeypatch):
    calls = []
    r = _router_with(lambda q: (calls.append(q), 8.0)[1], monkeypatch, llm_answer="Web_Search")
    assert r.route_query("what is the current status of shipping through the Strait of Hormuz?", "auto") == "Web_Search"
    assert calls == [], "the probe is not even consulted for a 'current' question"


def test_deterministic_rules_still_win_over_the_probe(monkeypatch):
    r = _router_with(lambda q: 9.0, monkeypatch)
    assert r.route_query("email Ali that the demo moved", "auto") == "Send_Email"


def test_probe_failure_or_none_is_harmless(monkeypatch):
    def boom(q): raise RuntimeError("index missing")
    r = _router_with(boom, monkeypatch, llm_answer="Direct_Chat")
    assert r.route_query("tell me something interesting about RAG systems", "auto") == "Direct_Chat"
    r = _router_with(lambda q: None, monkeypatch, llm_answer="Direct_Chat")
    assert r.route_query("tell me something interesting about RAG systems", "auto") == "Direct_Chat"


def test_threshold_is_the_measured_value():
    assert KB_PROBE_THRESHOLD == -3.0
