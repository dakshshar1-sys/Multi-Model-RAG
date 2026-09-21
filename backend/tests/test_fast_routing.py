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
