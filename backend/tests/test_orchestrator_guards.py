"""
Guards around the history-rewriter.

Background: with unrelated prior turns in the history, the 3B rewriter turned
"chart Samsung's 2025 revenue by quarter" into "What chart would you like to see
of **Samsung**'s projected revenue ...?" — a question back at the user — and that
string drove routing and search. The guard falls back to the user's own words.

Run:  cd backend && python -m pytest tests/test_orchestrator_guards.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The orchestrator imports the retrieval stack at module level, so even this pure-function
# test cannot be collected without the ML runtime. Skip the module, visibly, when it is absent.
pytest.importorskip("sentence_transformers", reason="needs the ML runtime; runs in the backend container")
pytest.importorskip("faiss", reason="needs the ML runtime; runs in the backend container")

from orchestrator.master_llm import looks_like_clarification  # noqa: E402

ORIG = "chart Samsung's 2025 revenue by quarter"


def test_question_back_at_the_user_is_caught():
    assert looks_like_clarification("What chart would you like to see for Samsung's projected revenue breakdown by quarter in 2025? Please provide any additional details or context if available.", ORIG)
    assert looks_like_clarification("Could you please specify which quarters you mean?", ORIG)
    assert looks_like_clarification("", ORIG)


def test_genuine_rewrites_pass():
    assert not looks_like_clarification("Samsung's 2025 revenue by quarter", ORIG)
    assert not looks_like_clarification("Samsung Electronics quarterly revenue for 2025, as a chart", ORIG)
    # a follow-up that was already a question stays a question
    assert not looks_like_clarification("What was Samsung's revenue in 2024?", "what about 2024?")


# ── Previous-image reuse ────────────────────────────────────────────────────
# Background: after one image upload, the old substring check ("the" in "there",
# "it" in "with") treated every later query as an image follow-up, so greetings
# and knowledge-base questions were answered from the stale picture and never
# reached the router. Found during the full feature check on 2026-09-14.

from orchestrator.master_llm import refers_to_previous_image


@pytest.mark.parametrize("q", [
    "hello there, what kinds of things are you able to help me with?",
    "look in my uploaded documents: which clustering algorithms do the ML Module-5 notes mention?",
    "search my uploaded documents: what does the ML MODULE-5 NOTES pdf say about clustering?",
    "chart Samsung's 2025 revenue by quarter",
    "make a bar chart: Jan 100, Feb 75, Mar 50",
    "it is important that we finish the report on time and send it to Ali",
    "write a script that reverses a string and save it as reverse.py",
])
def test_unrelated_queries_do_not_reuse_the_image(q):
    assert refers_to_previous_image(q) is False


@pytest.mark.parametrize("q", [
    "what is shown in this image?",
    "describe the photo",
    "read the text in the screenshot",
    "what's in it?",
    "can you zoom into that?",
    "summarize the attached image please",
])
def test_explicit_followups_do_reuse_the_image(q):
    assert refers_to_previous_image(q) is True
