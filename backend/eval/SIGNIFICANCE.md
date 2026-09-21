# Statistical significance of the evaluation (generated)

Produced by `python -m eval.significance` from the per-question results in `eval/results/`. No model is
called; nothing here is edited by hand. Method and formulas: the docstring of `eval/significance.py`.

## 1. Each configuration, with a 95% confidence interval

Wilson intervals. With 64 questions the interval is about ±0.11 wide, so two configurations can differ by
several points and still overlap: **overlap is not a test**, the paired comparison in section 2 is.

| configuration | correct | accuracy | 95% CI |
|---|---|---|---|
| no_retrieval | 2/64 | 0.031 | 0.009 – 0.107 |
| dense | 41/64 | 0.641 | 0.518 – 0.747 |
| dense+rerank | 44/64 | 0.688 | 0.566 – 0.788 |
| hybrid | 46/64 | 0.719 | 0.599 – 0.814 |
| hybrid+rerank | 47/64 | 0.734 | 0.615 – 0.827 |
| end-to-end, before retrieval-first routing | 6/64 | 0.094 | 0.044 – 0.190 |
| end-to-end, with retrieval-first routing | 43/64 | 0.672 | 0.550 – 0.774 |
| hybrid+rerank (llama3.2) | 42/64 | 0.656 | 0.534 – 0.761 |
| hybrid+rerank, greedy run 1 | 43/64 | 0.672 | 0.550 – 0.774 |
| retrieval recall@5: dense | 50/64 | 0.781 | 0.666 – 0.865 |
| retrieval recall@5: hybrid | 60/64 | 0.938 | 0.850 – 0.975 |

## 2. Paired comparisons (same questions, exact McNemar)

`gained` = questions B got right that A got wrong; `lost` = the reverse. Only those questions carry
information. `questions needed` = size of a question set that would confirm a gain of this size
(alpha 0.05, power 0.8).

| question | A → B | A | B | gained | lost | Δ | 95% CI of Δ | p | verdict | questions needed |
|---|---|---|---|---|---|---|---|---|---|---|
| is retrieval worth anything? | no_retrieval → dense | 0.031 | 0.641 | 39 | 0 | +0.609 | +0.500 – +0.734 | < 0.0001 | **significant** | – |
| the reranker, on dense | dense → dense+rerank | 0.641 | 0.688 | 9 | 6 | +0.047 | -0.078 – +0.172 | 0.607 | not significant at this n | ≈ 835 |
| lexical retrieval added to dense | dense → hybrid | 0.641 | 0.719 | 9 | 4 | +0.078 | -0.031 – +0.188 | 0.267 | not significant at this n | ≈ 259 |
| the reranker, on hybrid | hybrid → hybrid+rerank | 0.719 | 0.734 | 6 | 5 | +0.016 | -0.094 – +0.125 | 1.000 | not significant at this n | ≈ 5524 |
| the whole shipped retrieval stack vs baseline RAG | dense → hybrid+rerank | 0.641 | 0.734 | 13 | 7 | +0.094 | -0.047 – +0.234 | 0.263 | not significant at this n | ≈ 277 |
| the routing fix, end to end | end-to-end, before retrieval-first routing → end-to-end, with retrieval-first routing | 0.094 | 0.672 | 38 | 1 | +0.578 | +0.453 – +0.703 | < 0.0001 | **significant** | – |
| qwen2.5:3b vs llama3.2, same stack | hybrid+rerank → hybrid+rerank (llama3.2) | 0.734 | 0.656 | 6 | 11 | -0.078 | -0.203 – +0.047 | 0.332 | not significant at this n | ≈ 340 |
| hybrid retrieval, measured at retrieval (no generation noise) | retrieval recall@5: dense → retrieval recall@5: hybrid | 0.781 | 0.938 | 10 | 0 | +0.156 | +0.078 – +0.250 | 0.0020 | **significant** | – |
| does greedy decoding (temperature 0) cost accuracy? | hybrid+rerank → hybrid+rerank, greedy run 1 | 0.734 | 0.672 | 4 | 8 | -0.062 | -0.172 – +0.047 | 0.388 | not significant at this n | ≈ 375 |

## 3. The noise floor: the identical configuration, run repeatedly

Same retrieval, same prompt, same model. Any difference here is noise, and it is the yardstick for
every generation-level difference above.

### Sampled decoding (the model's default temperature, what production uses)

| run A → run B | A | B | verdicts that flipped | identical answers | p |
|---|---|---|---|---|---|
| hybrid+rerank → hybrid+rerank, repeat run 2 | 0.734 | 0.734 | 8/64 (12.5%) | 4.7% | 1.000 |
| hybrid+rerank → hybrid+rerank, repeat run 3 | 0.734 | 0.703 | 10/64 (15.6%) | 1.6% | 0.754 |
| hybrid+rerank, repeat run 2 → hybrid+rerank, repeat run 3 | 0.734 | 0.703 | 10/64 (15.6%) | 1.6% | 0.754 |

Accuracy ranged 0.703–0.734 with nothing changed, and on average **14.6% of questions flipped verdict between two identical runs**. None of these pairs is significant, as it should be: the test does not mistake noise for an effect. But compare the flip counts with the `gained + lost` counts in section 2: most single-run answer-level differences there are no larger than this.

### Greedy decoding (`OLLAMA_TEMPERATURE=0`, for evaluation runs)

| run A → run B | A | B | verdicts that flipped | identical answers | p |
|---|---|---|---|---|---|
| hybrid+rerank, greedy run 1 → hybrid+rerank, greedy run 2 | 0.672 | 0.688 | 1/64 (1.6%) | 87.5% | 1.000 |

With greedy decoding 1 of 64 verdicts flipped (1.6%, against 14.6% sampled) and 87.5% of answers were character-for-character identical. Evaluation runs should set `OLLAMA_TEMPERATURE=0`: a difference between two greedy runs is then a difference between two systems. Production keeps the model's default sampling; this is an evaluation setting.

## 4. What can and cannot be claimed

**Supported by this evaluation (p < 0.05):**

- is retrieval worth anything?: 0.031 → 0.641 (39 gained, 0 lost, p < 0.0001).
- the routing fix, end to end: 0.094 → 0.672 (38 gained, 1 lost, p < 0.0001).
- hybrid retrieval, measured at retrieval (no generation noise): 0.781 → 0.938 (10 gained, 0 lost, p 0.0020).

**Not established at n = 64, in either direction:**

- the reranker, on dense: +0.047 (9 gained, 6 lost, p 0.607); a set of about 835 questions would settle it.
- lexical retrieval added to dense: +0.078 (9 gained, 4 lost, p 0.267); a set of about 259 questions would settle it.
- the reranker, on hybrid: +0.016 (6 gained, 5 lost, p 1.000); a set of about 5524 questions would settle it.
- the whole shipped retrieval stack vs baseline RAG: +0.094 (13 gained, 7 lost, p 0.263); a set of about 277 questions would settle it.
- qwen2.5:3b vs llama3.2, same stack: -0.078 (6 gained, 11 lost, p 0.332); a set of about 340 questions would settle it.
- does greedy decoding (temperature 0) cost accuracy?: -0.062 (4 gained, 8 lost, p 0.388); a set of about 375 questions would settle it.

The honest wording for these is "consistent with a difference", not "differs". The remedy is more
questions, which is what `datasets/INDEPENDENT_PROTOCOL.md` is for; the `questions needed` column says how many.

Retrieval-level metrics (recall@k) involve no sampled generation, so they are the cleaner place to
demonstrate a retrieval change; answer-level metrics inherit the noise in section 3.
