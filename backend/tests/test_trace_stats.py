"""Latency statistics over request traces: percentiles, per-tool, per-stage, rates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trace_stats import format_markdown, percentile, summarize


def _t(tool, total, stages=None, conv="ui", **flags):
    return {"tool": tool, "total_ms": total, "ms_by_model": stages or {}, "conversation_id": conv,
            "started_at": "2026-09-21T10:00:00.000+00:00", **flags}


def test_percentile_interpolates_and_handles_edges():
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 90) == 46.0
    assert percentile([5, 1, 3], 0) == 1.0 and percentile([5, 1, 3], 100) == 5.0


def test_summary_groups_by_tool_and_stage_and_excludes_cache_hits_from_timing():
    traces = [_t("Search_Knowledge_Base", 10000, {"Generation": 6000, "Reranking Model": 1000}, verification="pass"),
              _t("Search_Knowledge_Base", 14000, {"Generation": 9000, "Reranking Model": 1000}, verification="flagged"),
              _t("Web_Search", 26000, {"Web Search": 9000, "Generation": 8000}, search_degraded=True, chart=True),
              _t(None, 4, cache_hit=True)]
    s = summarize(traces)
    assert s["n"] == 4 and s["cache_hit_rate"] == 0.25 and s["search_degraded_rate"] == 0.25 and s["chart_rate"] == 0.25
    assert s["overall"]["n"] == 3, "a 4 ms cache hit must not drag the latency distribution down"
    assert s["by_tool"]["Search_Knowledge_Base"] == {"n": 2, "mean_ms": 12000, "max_ms": 14000, "p50_ms": 12000, "p90_ms": 13600, "p95_ms": 13800}
    assert s["by_tool"]["(cache hit)"]["n"] == 1
    gen = s["by_stage"]["Generation"]
    assert gen["n"] == 3 and gen["share_of_uncached_time"] == round(23000 / 50000, 3)
    assert list(s["by_stage"])[0] == "Generation", "stages are ordered by share of time"
    assert s["verification"] == {"pass": 1, "flagged": 1, "not run": 2}
    assert s["abstained_rate"] == 0.0
    assert summarize([_t("Search_Knowledge_Base", 900, abstained=True), _t("Search_Knowledge_Base", 9000)])["abstained_rate"] == 0.5


def test_conversation_filter_and_empty_input():
    traces = [_t("Direct_Chat", 1000, conv="a"), _t("Direct_Chat", 3000, conv="b")]
    assert summarize(traces, conversation="a")["overall"]["p50_ms"] == 1000
    assert summarize(traces, conversation="zzz") == {"n": 0}
    assert summarize([]) == {"n": 0} and summarize([{"junk": 1}, "not a dict"]) == {"n": 0}


def test_markdown_has_both_tables_and_survives_no_data():
    md = format_markdown(summarize([_t("Web_Search", 26000, {"Web Search": 9000})]))
    assert "## End-to-end time by tool" in md and "| Web_Search | 1 | 26.0 |" in md
    assert "## Time by pipeline stage" in md and "| Web Search | 1 | 9.0 |" in md
    assert "No traces recorded yet" in format_markdown({"n": 0})
