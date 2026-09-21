"""
Per-request tracing built from the pipeline's own stage events.

The orchestrator already narrates every step as SSE events ("Web Search",
"Processing" -> "Completed"). Recording those events with a clock gives, for
free, what a results chapter and a slow-demo diagnosis both need: how long each
stage took, which tool was chosen, whether the cache answered, whether search
was degraded, and the total. Nothing here touches the pipeline's logic; it only
watches the events go by.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_TOOL_RE = re.compile(r"Selected Tool: \[([A-Za-z_]+)\]")
_STAGE_FIELDS = ("model", "status", "action")


class RequestTrace:
    def __init__(self, query: str, conversation_id: str | None = None, model_choice: str = "auto"):
        self.request_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()
        self.query = query
        self.conversation_id = conversation_id or "default"
        self.model_choice = model_choice
        self.events: list[dict] = []
        self._open: dict[str, float] = {}        # model -> t of its Processing event
        self.stages: list[dict] = []             # completed stages with durations
        self.tool: str | None = None
        self.cache_hit = False
        self.search_degraded = False
        self.chart = False
        self.verification: str | None = None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def record(self, event: dict) -> int:
        """Observe one emitted event; returns elapsed ms (stamped onto the event by the caller)."""
        t = time.perf_counter()
        ms = int((t - self._t0) * 1000)
        model, status, action = (event.get(k) or "" for k in _STAGE_FIELDS)
        if action == "Streaming":
            return ms                                  # token chunks are not stages
        self.events.append({"t_ms": ms, "model": model, "status": status, "action": action[:160]})
        if status == "Processing":
            self._open.setdefault(model, t)
        elif status == "Completed":
            start = self._open.pop(model, None)
            self.stages.append({"model": model, "action": action[:160],
                                "ms": int((t - start) * 1000) if start is not None else 0, "t_ms": ms})
        m = _TOOL_RE.search(action)
        if m:
            self.tool = m.group(1)
        if model == "Cache Manager":
            self.cache_hit = True
        if "rate-limiting this address" in action:
            self.search_degraded = True
        if model == "Verification Module" and status == "Completed":
            self.verification = "pass" if "passed" in action.lower() else "flagged"
        details = event.get("details") or {}
        if details.get("chart"):
            self.chart = True
        return ms

    def summary(self) -> dict:
        by_model: dict[str, int] = {}
        for s in self.stages:
            by_model[s["model"]] = by_model.get(s["model"], 0) + s["ms"]
        return {
            "request_id": self.request_id, "started_at": self.started_at,
            "conversation_id": self.conversation_id, "model_choice": self.model_choice,
            "query": self.query[:200], "total_ms": self.elapsed_ms(),
            "tool": self.tool, "cache_hit": self.cache_hit, "search_degraded": self.search_degraded,
            "chart": self.chart, "verification": self.verification,
            "retrieval_mode": "hybrid" if os.getenv("HYBRID_RETRIEVAL", "1").lower() not in ("0", "false", "no") else "dense",
            "stages": self.stages, "ms_by_model": by_model,
        }


class TraceLog:
    """Append-only JSON lines of trace summaries, with a bounded tail reader."""
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(os.getenv("DATA_DIR", "data"), "traces.jsonl")
        self._lock = threading.Lock()

    def append(self, summary: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Trace log write failed: {e}")

    def tail(self, limit: int = 50) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
            return [json.loads(l) for l in lines if l.strip()]
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning(f"Trace log read failed: {e}")
            return []
