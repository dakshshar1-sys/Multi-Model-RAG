"""
Latency report from the accumulated request traces.

    cd backend && python -m eval.latency_report                      # -> eval/LATENCY_REPORT.md
    python -m eval.latency_report --conversation ablation           # only harness-driven requests
    python -m eval.latency_report --json                            # print the summary as JSON

Closes the "latency distribution over many requests" item in BASELINE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.request_trace import TraceLog
from core.trace_stats import format_markdown, summarize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LATENCY_REPORT.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=None, help="path to traces.jsonl (default: $DATA_DIR/traces.jsonl)")
    ap.add_argument("--conversation", default=None, help="restrict to one conversation_id")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    summary = summarize(TraceLog(args.traces).tail(args.limit), args.conversation)
    if args.json:
        print(json.dumps(summary, indent=2)); return
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(format_markdown(summary))
    print(f"wrote {OUT} ({summary.get('n', 0)} traces)")


if __name__ == "__main__":
    main()
