"""Request tracing is derived purely from the pipeline's own events."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.request_trace import RequestTrace, TraceLog


def _ev(model, status, action, details=None):
    d = {"model": model, "status": status, "action": action}
    if details: d["details"] = details
    return d


def test_stage_durations_tool_flags_and_summary():
    tr = RequestTrace("chart Samsung's 2025 revenue by quarter", "conv-A", "auto")
    tr.record(_ev("Agent Router", "Processing", "Classifying intent..."))
    time.sleep(0.02)
    tr.record(_ev("Agent Router", "Completed", "Selected Tool: [Visualize_Data]"))
    tr.record(_ev("Web Search", "Processing", "Initiating multi-path research"))
    tr.record(_ev("Web Search", "Completed", "Primary search engine is rate-limiting this address — results below are from fallback sources"))
    tr.record(_ev("Final Response", "Processing", "Streaming"))        # token chunk: ignored
    tr.record(_ev("Verification Module", "Completed", "Response passed factuality check: ok"))
    tr.record(_ev("Final Response", "Completed", "Pipeline finished", {"answer": "x", "chart": "/uploads/c.png"}))
    s = tr.summary()
    assert s["tool"] == "Visualize_Data" and s["search_degraded"] is True and s["chart"] is True
    assert s["verification"] == "pass" and s["cache_hit"] is False
    stages = {st["model"]: st for st in s["stages"]}
    assert stages["Agent Router"]["ms"] >= 20, "duration spans Processing -> Completed"
    assert stages["Web Search"]["ms"] >= 0
    assert "Streaming" not in [e["action"] for e in tr.events]
    assert s["total_ms"] >= stages["Agent Router"]["ms"]
    assert s["conversation_id"] == "conv-A" and s["retrieval_mode"] in ("hybrid", "dense")


def test_cache_hit_is_detected_and_completed_without_processing_has_zero_duration():
    tr = RequestTrace("q")
    tr.record(_ev("Cache Manager", "Completed", "Cache hit! Retrieved answer instantly."))
    s = tr.summary()
    assert s["cache_hit"] is True and s["stages"][0]["ms"] == 0 and s["tool"] is None


def test_trace_log_appends_and_tails(tmp_path):
    log = TraceLog(path=str(tmp_path / "t" / "traces.jsonl"))
    assert log.tail() == []
    for i in range(5):
        log.append({"request_id": str(i), "total_ms": i})
    assert [t["request_id"] for t in log.tail(3)] == ["2", "3", "4"]
    assert len(log.tail(100)) == 5
    with open(log.path) as f:
        assert all(json.loads(l) for l in f)


def test_a_declined_request_is_recorded_as_abstained():
    tr = RequestTrace("in my documents, what is the boiling point of ethanol?")
    tr.record(_ev("Reranking Model", "Processing", "Cross-encoding"))
    tr.record(_ev("Reranking Model", "Completed", "No passage is relevant enough (best -11.1, floor -5.0) — declining rather than guessing"))
    tr.record(_ev("Final Response", "Completed", "Declined: not in the knowledge base", {"answer": "I couldn't find anything", "sources": [], "abstained": True}))
    assert tr.summary()["abstained"] is True
    assert RequestTrace("q").summary()["abstained"] is False
