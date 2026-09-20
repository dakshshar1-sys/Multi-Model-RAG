# Baseline results — 2026-09-20

All numbers from `eval/run_eval.py` on the local stack (qwen2.5:3b via Ollama, all-MiniLM-L6-v2
embeddings, ms-marco-MiniLM-L-6-v2 cross-encoder, RTX 2050 4 GB). Raw per-item rows are in the
JSON files these lines came from; `results/history.md` accumulates every run.

## Routing (32 cases, 3 runs each)

| session | strict accuracy (mean) | effective | majority | unstable cases | s/case |
|---|---|---|---|---|---|
| 1 | 0.854 | – | 0.875 | 2 (r06, r25) | 1.81 |
| 2 | 0.906 | 0.938 | 0.906 | 0 | 0.45 |

The spread between sessions is real: the classifier is a sampled 3B model. Report both.
Confusions in session 1: Send_Email→Send_Telegram ×4, Search_Knowledge_Base→Web_Search ×4,
Web_Search→Visualize_Data ×3 (corrected downstream by the no-numbers guard, hence "effective"),
Ambiguous_Query→Web_Search ×3.

## Retrieval (64 questions, evidence-substring hits)

| configuration | recall@5 | recall@10 | MRR |
|---|---|---|---|
| dense only (FAISS top-10, no rerank) | 0.781 | 0.922 | 0.681 |
| dense + cross-encoder rerank to top-5 | **0.891** | 0.922 | **0.831** |

The reranker pulls 9 items into the top-5 and pushes 2 out (q23, q42): net +7, +0.11 recall@5.

Five questions miss the top-10 entirely (q04, q06, q07, q20, q36). Every one is a short
bulleted line whose evidence sits in exactly one chunk. Dense retrieval ranks those chunks
12th–36th of 79; a plain BM25 over the same chunks ranks four of them 1st or 2nd.

| top-5 hit over all 64 | count |
|---|---|
| dense | 50 |
| BM25 | 60 |
| union of the two top-5 lists | 62 |

**This is the measured case for hybrid retrieval (BM25 + dense with rank fusion).**

## Answers (64 questions, top-5 reranked context, LLM judge on)

| metric | value |
|---|---|
| contains-gold (≥80% of gold content tokens present) | 0.625 |
| contains-gold **given the evidence was in context** | 0.702 |
| token-F1 vs gold | 0.327 (long analytical prose vs short gold; expected low) |
| lexical support (model-free faithfulness proxy) | 0.778 |
| judge PASS rate (existing verifier as judge) | 0.906 |
| generation time | 2.86 s/answer |

Two findings to write up:

1. **Generation loses the fact ~30% of the time even when retrieval delivered it.** 17 of the
   24 contains-gold failures had the evidence in context: q13, q14, q22, q25, q26, q27, q30, q31, q32, q33, q44, q46, q50, q51, q56, q58, q62. The 8-item smoke run
   showed 1.0 here; the full run says 0.70. Small samples flatter.
2. **The LLM judge is lenient.** It passes 0.906 of answers while only 0.625 contain the gold fact,
   and it disagrees with the lexical proxy on 17 items: q06, q07, q14, q18, q19, q22, q23, q31, q33, q34, q35, q40, q42, q45, q59, q60, q64. Read those by hand;
   they are the material for a section on why a 3B verifier is not a reliable judge.

## Not yet measured

- Hybrid retrieval (next change; the diagnostic above predicts recall@5 ≈ 0.95+).
- OCR ingestion (added the same day; measure with a scanned-PDF corpus once one exists).
- `qa.jsonl` items are still `reviewed: false`; re-run after review and record the delta.
