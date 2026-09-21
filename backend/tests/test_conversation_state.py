"""
Per-conversation state must not leak between conversations.

Background: the orchestrator held "the last uploaded image" as one attribute on a
process-wide singleton, so an image uploaded in one tab could answer a follow-up
in another. Found during the full feature check on 2026-09-14; fixed 2026-09-21.

Run:  cd backend && python -m pytest tests/test_conversation_state.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_state import DEFAULT_ID, ConversationStore


def test_states_are_isolated_per_conversation():
    store = ConversationStore()
    a, b = store.get("conv-a"), store.get("conv-b")
    a.set_image("a picture of a cat")
    assert store.get("conv-a").last_image_context == "a picture of a cat"
    assert store.get("conv-b").last_image_context is None, "another conversation must not see it"
    assert a is store.get("conv-a") and a is not b


def test_missing_id_uses_the_default_bucket():
    store = ConversationStore()
    assert store.get(None).conversation_id == DEFAULT_ID
    assert store.get("").conversation_id == DEFAULT_ID
    assert store.get("   ").conversation_id == DEFAULT_ID
    store.get(None).set_image("x")
    assert store.get("").last_image_context == "x", "all id-less requests share one bucket (old behaviour)"


def test_image_age_and_clear():
    st = ConversationStore().get("c")
    assert st.image_age_s() == float("inf")
    st.set_image("img")
    assert 0.0 <= st.image_age_s() < 1.0
    st.clear_image()
    assert st.last_image_context is None and st.image_age_s() == float("inf")


def test_idle_conversations_are_evicted(monkeypatch):
    store = ConversationStore(idle_ttl_s=100.0)
    import core.conversation_state as cs
    t = [1000.0]
    monkeypatch.setattr(cs.time, "monotonic", lambda: t[0])
    store.get("old").set_image("img")
    t[0] += 50; store.get("fresh")
    t[0] += 60                     # "old" idle 110s, "fresh" idle 60s
    store.get("probe")
    assert len(store) == 2 and store.get("fresh") is not None
    assert store.get("old").last_image_context is None, "recreated empty after eviction"


def test_cap_evicts_least_recently_used():
    store = ConversationStore(max_conversations=3)
    for cid in ("a", "b", "c"):
        store.get(cid)
    store.get("a")          # a is now most recent; b is least
    store.get("d")          # over cap -> evict b
    assert len(store) == 3
    assert set(store._states) == {"a", "c", "d"}


@pytest.mark.needs_ml
def test_orchestrator_signature_accepts_conversation_id():
    import inspect
    import orchestrator.master_llm as m
    cls = next(v for k, v in vars(m).items() if inspect.isclass(v) and hasattr(v, "process_query_stream"))
    assert "conversation_id" in inspect.signature(cls.process_query_stream).parameters
    assert not hasattr(cls, "last_image_context")
