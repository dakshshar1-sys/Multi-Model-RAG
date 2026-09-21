"""
Latency statistics over recorded request traces (see core/request_trace.py).

One traced request is an anecdote; this turns the accumulated traces.jsonl into a
distribution: p50/p90/p95 of end-to-end time per tool, the same per pipeline
stage with each stage's share of the total, and the rates that explain the
spread (cache hits, degraded searches, verification verdicts). Pure functions
over a list of trace dicts, so the API endpoint, the report CLI and the tests
all share one implementation.
"""
from __future__ import annotations

import math
from collections import defaultdict

PERCENTILES = (50, 90, 95)


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (the 'inclusive' method); None for no data."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def _dist(values: list[float]) -> dict:
    out = {"n": len(values)}
    if values:
        out["mean_ms"] = round(sum(values) / len(values))
        out["max_ms"] = round(max(values))
        for p in PERCENTILES:
            out[f"p{p}_ms"] = round(percentile(values, p))
    return out


def _tool_of(trace: dict) -> str:
    if trace.get("cache_hit"):
        return "(cache hit)"
    return trace.get("tool") or "(no tool)"


def summarize(traces: list[dict], conversation: str | None = None) -> dict:
    """
    Aggregate traces. `conversation` restricts to one conversation_id (e.g. "ablation"
    to look only at harness-driven requests, or a UI conversation to debug one session).
    """
    rows = [t for t in traces if isinstance(t, dict) and isinstance(t.get("total_ms"), (int, float))]
    if conversation is not None:
        rows = [t for t in rows if t.get("conversation_id") == conversation]
    if not rows:
        return {"n": 0}

    by_tool: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    stage_total: dict[str, float] = defaultdict(float)
    verification = defaultdict(int)
    uncached = [t for t in rows if not t.get("cache_hit")]
    for t in rows:
        by_tool[_tool_of(t)].append(float(t["total_ms"]))
        verification[t.get("verification") or "not run"] += 1
    for t in uncached:
        for model, ms in (t.get("ms_by_model") or {}).items():
            by_stage[model].append(float(ms))
            stage_total[model] += float(ms)

    grand = sum(float(t["total_ms"]) for t in uncached) or 1.0
    stages = {}
    for model, vals in by_stage.items():
        d = _dist(vals)
        d["share_of_uncached_time"] = round(stage_total[model] / grand, 3)
        stages[model] = d
    started = sorted(t.get("started_at") or "" for t in rows)
    return {
        "n": len(rows),
        "window": {"first": started[0], "last": started[-1]},
        "overall": _dist([float(t["total_ms"]) for t in uncached]),
        "cache_hit_rate": round(sum(bool(t.get("cache_hit")) for t in rows) / len(rows), 3),
        "search_degraded_rate": round(sum(bool(t.get("search_degraded")) for t in rows) / len(rows), 3),
        "chart_rate": round(sum(bool(t.get("chart")) for t in rows) / len(rows), 3),
        "abstained_rate": round(sum(bool(t.get("abstained")) for t in rows) / len(rows), 3),
        "verification": dict(verification),
        "by_tool": {k: _dist(v) for k, v in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))},
        "by_stage": dict(sorted(stages.items(), key=lambda kv: -kv[1]["share_of_uncached_time"])),
        "conversations": dict(sorted(_count(rows, "conversation_id").items(), key=lambda kv: -kv[1])),
    }


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[str(r.get(key))] += 1
    return dict(out)


def _s(ms) -> str:
    return "–" if ms is None else f"{ms / 1000:.1f}"


def format_markdown(summary: dict, title: str = "Latency report (generated)") -> str:
    if not summary.get("n"):
        return f"# {title}\n\nNo traces recorded yet. Send some requests, then re-run.\n"
    o = summary["overall"]
    lines = [f"# {title}\n",
             "Produced by `python -m eval.latency_report` from `data/traces.jsonl`, which the orchestrator",
             "appends to on every request. Times are wall-clock seconds on the development machine",
             "(3B local model on a 4 GB GPU; reranker, embeddings and OCR on CPU). Nothing is edited by hand.\n",
             f"- **{summary['n']} requests**, {summary['window']['first'][:19]} → {summary['window']['last'][:19]} UTC",
             f"- End to end, uncached: **p50 {_s(o.get('p50_ms'))} s · p90 {_s(o.get('p90_ms'))} s · p95 {_s(o.get('p95_ms'))} s** · max {_s(o.get('max_ms'))} s",
             f"- Cache hits {summary['cache_hit_rate']:.1%} · degraded searches {summary['search_degraded_rate']:.1%} · answers with a chart {summary['chart_rate']:.1%} · declined (not in the knowledge base) {summary.get('abstained_rate', 0):.1%}",
             f"- Verification verdicts: {', '.join(f'{k} {v}' for k, v in summary['verification'].items())}",
             f"- Request sources (conversation id → count): {', '.join(f'{k} {v}' for k, v in summary['conversations'].items())}\n",
             "## End-to-end time by tool\n",
             "| tool | n | p50 s | p90 s | p95 s | mean s | max s |", "|---|---|---|---|---|---|---|"]
    for tool, d in summary["by_tool"].items():
        lines.append(f"| {tool} | {d['n']} | {_s(d.get('p50_ms'))} | {_s(d.get('p90_ms'))} | {_s(d.get('p95_ms'))} | {_s(d.get('mean_ms'))} | {_s(d.get('max_ms'))} |")
    lines += ["\n## Time by pipeline stage (uncached requests)\n",
              "`share` is the stage's fraction of all uncached wall-clock time. Shares do not sum to 1: time",
              "between stages (history rewrite, prompt assembly, streaming) is not inside any stage.\n",
              "| stage | n | p50 s | p90 s | p95 s | share |", "|---|---|---|---|---|---|"]
    for stage, d in summary["by_stage"].items():
        lines.append(f"| {stage} | {d['n']} | {_s(d.get('p50_ms'))} | {_s(d.get('p90_ms'))} | {_s(d.get('p95_ms'))} | {d['share_of_uncached_time']:.1%} |")
    return "\n".join(lines) + "\n"
