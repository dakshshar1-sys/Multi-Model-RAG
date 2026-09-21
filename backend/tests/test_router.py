"""
Tests for the router's deterministic fast-path.

Since 2026-09-21 the fast path also routes other unmistakable intents (sends, inbox
reads, own-document questions, inline charts, greetings, fragments); the invariant
these tests guard is unchanged: a non-file request must never become a file write.

The 3B local classifier occasionally misrouted an unmistakable "make a file/folder"
request to a messaging tool (a folder+file prompt once produced a WhatsApp draft). The
fast-path decides those clear cases with regex before the LLM. These tests pin that it
fires for real file/folder requests and — critically — does NOT hijack genuine sends.

Run:  cd backend && source venv/bin/activate && python -m pytest tests/test_router.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agentic_router import AgentRouter


@pytest.mark.parametrize("query", [
    "make a folder named pythoncode and in that folder make a python file named rng.py "
    "that generates and prints a random number",
    "write a script that reverses a string and save it as reverse.py",
    "create index.html with a hello world page",
    "make a folder called reports",
    "generate a config.yaml for the service",
    "build a program in a folder called src",
])
def test_fast_path_forces_workspace_for_file_requests(query):
    assert AgentRouter._fast_route(query) == "Workspace_Task"


@pytest.mark.parametrize("query", [
    "email Ali the config.py file",          # a send that mentions a file — must not be forced
    "telegram Ishan that the build passed",
    "send Ali a message about main.py",
    "whatsapp mom the recipe",
])
def test_fast_path_never_hijacks_a_send(query):
    # Any messaging cue backs off to the LLM, so genuine sends still route to messaging.
    assert AgentRouter._fast_route(query) != "Workspace_Task"


@pytest.mark.parametrize("query", [
    "what is retrieval augmented generation",
    "make a bar chart: apples 40, bananas 65",   # visualization, not a file
    "summarize the document I uploaded",
    "who won the match yesterday",
])
def test_fast_path_stays_out_of_non_file_queries(query):
    assert AgentRouter._fast_route(query) != "Workspace_Task"
