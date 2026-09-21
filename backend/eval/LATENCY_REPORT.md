# Latency report (generated)

Produced by `python -m eval.latency_report` from `data/traces.jsonl`, which the orchestrator
appends to on every request. Times are wall-clock seconds on the development machine
(3B local model on a 4 GB GPU; reranker, embeddings and OCR on CPU). Nothing is edited by hand.

- **146 requests**, 2026-09-20T18:57:29 → 2026-09-21T11:02:12 UTC
- End to end, uncached: **p50 10.6 s · p90 21.8 s · p95 25.5 s** · max 80.9 s
- Cache hits 0.7% · degraded searches 0.0% · answers with a chart 3.4%
- Verification verdicts: not run 13, flagged 37, pass 96
- Request sources (conversation id → count): ablation 128, profile 6, profile2 6, probe-check 3, trace-demo 2, probe-email 1

## End-to-end time by tool

| tool | n | p50 s | p90 s | p95 s | mean s | max s |
|---|---|---|---|---|---|---|
| Search_Knowledge_Base | 67 | 4.3 | 8.9 | 11.5 | 5.2 | 16.5 |
| Web_Search | 64 | 18.7 | 24.8 | 30.7 | 19.2 | 42.1 |
| Visualize_Data | 4 | 15.5 | 64.4 | 72.6 | 28.9 | 80.9 |
| Send_Email | 3 | 3.4 | 3.6 | 3.6 | 2.3 | 3.6 |
| Direct_Chat | 2 | 5.8 | 9.7 | 10.1 | 5.8 | 10.6 |
| Workspace_Task | 2 | 2.3 | 2.9 | 2.9 | 2.3 | 3.0 |
| Read_Email | 2 | 14.5 | 14.8 | 14.8 | 14.5 | 14.9 |
| (cache hit) | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ambiguous_Query | 1 | 2.5 | 2.5 | 2.5 | 2.5 | 2.5 |

## Time by pipeline stage (uncached requests)

`share` is the stage's fraction of all uncached wall-clock time. Shares do not sum to 1: time
between stages (history rewrite, prompt assembly, streaming) is not inside any stage.

| stage | n | p50 s | p90 s | p95 s | share |
|---|---|---|---|---|---|
| Generation | 135 | 5.2 | 9.0 | 10.9 | 41.4% |
| Web Search | 66 | 7.7 | 12.6 | 14.8 | 32.9% |
| Agent Router | 145 | 0.9 | 1.1 | 1.1 | 6.2% |
| Reranking Model | 67 | 1.0 | 1.3 | 1.4 | 3.7% |
| Email Agent | 5 | 2.8 | 6.1 | 6.2 | 1.0% |
| Visualizer Agent | 98 | 0.0 | 0.0 | 0.0 | 0.5% |
| Workspace Agent | 2 | 2.3 | 2.9 | 2.9 | 0.3% |
| Vector Retrieval | 67 | 0.0 | 0.1 | 0.1 | 0.2% |
| Direct Chat | 2 | 0.9 | 1.0 | 1.0 | 0.1% |
| Master LLM Orchestrator | 145 | 0.0 | 0.0 | 0.0 | 0.0% |
| Embedding Model | 67 | 0.0 | 0.0 | 0.0 | 0.0% |
| Verification Module | 133 | 0.0 | 0.0 | 0.0 | 0.0% |
| Final Response | 145 | 0.0 | 0.0 | 0.0 | 0.0% |
