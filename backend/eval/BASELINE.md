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

## Hybrid retrieval — measured 2026-09-20 (same day, after the diagnostic above)

BM25 (dependency-free, `retrieval/bm25.py`) over the same chunks, fused with dense by
reciprocal rank fusion (k=60, candidate pools of 20 each), then the same cross-encoder.

| configuration | recall@5 | recall@10 | MRR | misses (top-10) |
|---|---|---|---|---|
| dense only | 0.781 | 0.922 | 0.681 | 5 |
| dense + rerank (previous default) | 0.891 | 0.922 | 0.831 | 5 |
| **hybrid, no rerank** | **0.938** | **0.984** | 0.803 | 1 (q20) |
| hybrid + rerank (new default) | 0.922 | 0.984 | **0.866** | 1 |
| BM25 only, no rerank | 0.969 | 1.000 | 0.890 | 0 |

Three things to write up honestly:

1. **Hybrid delivers what the diagnostic predicted**: recall@5 0.781 → 0.938, recall@10 0.922 → 0.984.
2. **After hybrid, the reranker slightly lowers recall@5** (0.938 → 0.922; pushed out ['q23', 'q42', 'q56'], pulled in ['q06', 'q24'])
   while still raising MRR (0.803 → 0.866). It reorders well but its top-5 cut costs a hit. Options to
   test: pass top-6/7 after reranking, or rerank only when the fused pool disagrees.
3. **BM25 alone scores highest on this set (0.969).** That is a property of the questions, not the method:
   they were written by reading the documents and reuse their vocabulary, which favours lexical matching.
   Real users paraphrase, which favours dense. This is why the default is hybrid, not lexical-only, and it is
   the strongest argument for adding a paraphrased question set (or questions from someone who has not read
   the corpus) before drawing conclusions about dense vs lexical.

## Not yet measured

- OCR ingestion (added the same day; measure with a scanned-PDF corpus once one exists).
- `qa.jsonl` items are still `reviewed: false`; re-run after review and record the delta.
