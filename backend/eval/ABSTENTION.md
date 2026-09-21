# Abstention: does the system say "not in your documents"?

*2026-09-21 · `python -m eval.abstention` · qwen2.5:3b · results in `eval/results/abstention_*.json`*

Every other suite in this harness asks questions the corpus can answer, so a system that
always answers looks perfect on them. This one asks 24 questions the corpus **cannot** answer
and counts how often the knowledge-base path declines instead of inventing.

## The questions

`datasets/unanswerable.jsonl`: 20 **near** questions, phrased like the real ones and on-topic
("What BLEU score does the report give for the generation model?", "Who is named as the
project supervisor?"), and 4 **far** ones ("What is the capital city of Mongolia?"). Each line
lists the terms that make it unanswerable, and `tests/test_abstention.py` fails if any of
those terms ever appears in the corpus, so the set cannot silently become answerable.

The 64 answerable questions run through the same path to price the other side: declining
when the answer was there.

## What was measured

Two defences, separately, because they fail differently:

- **generator**: given weak context, the model itself says the documents do not contain it.
- **answer floor**: the best cross-encoder score among the retrieved passages is below −5.0,
  so nothing retrieved is about the question, and no answer is generated at all.

| | before (generator only) | after (generator + floor at −5.0) |
|---|---|---|
| unanswerable declined | 21–22 of 24 (0.88–0.92) | **24 of 24 (1.000)** |
| — near | 19–20 of 20 | 20 of 20 |
| — far | 2 of 4 | 4 of 4 |
| answerable wrongly declined | 2 of 64 (0.031) | 2 of 64 (0.031) |
| time for a declined request | 3.2 s (a full generation) | 0.5 s (no generation) |

Median best passage score: **−8.9** for unanswerable questions, **+4.4** for answerable ones.

## What the generator gets wrong, and why it matters

The model almost never invents a *document* fact: asked for a BLEU score or a budget that
is not there, it says so. What it does instead is answer **from its own memory and cite the
documents for it**:

> Q: What is the boiling point of ethanol in degrees Celsius?
> A: The boiling point of ethanol in degrees Celsius is 78.4°C **[1]**.

> Q: What is the capital city of Mongolia?
> A: The capital city of Mongolia is Ulaanbaatar. **[1]**

Both facts are true. Source [1] is a chapter about this project's API. A correct answer
with a false citation is worse than it looks: it passes a fact-check, and it teaches the
user that `[1]` means "verified in your documents" when it does not. These score −11; the
floor stops them. The third miss was a false-premise question ("Why did the authors choose
Weaviate over FAISS?"), which drew a confused non-answer; it scored −6.7 and the floor
stops it too.

## Choosing the floor

Best-score thresholds, swept offline over the recorded scores (final run):

| floor | unanswerable stopped by the floor alone | declined by floor or generator | answerable stopped | …of which had been answered correctly |
|---|---|---|---|---|
| −8 | 14/24 | 23/24 | 0/64 | 0 |
| −6 | 19/24 | 24/24 | 0/64 | 0 |
| **−5** | **22/24** | **24/24** | **1/64** | **0** |
| −4 | 23/24 | 24/24 | 2/64 | 0 |
| −2 | 23/24 | 24/24 | 5/64 | 1 |
| 0 | 24/24 | 24/24 | 8/64 | 2 |

−5.0 was chosen because it is not a new number: it is the reranker's existing relevance
floor. The reranker already logged *"all documents scored below threshold"* for these
queries and then passed its top result on regardless. The change makes the pipeline act on
a verdict it was already reaching. At −5 the one answerable question it stops (q36, −6.0)
was answered wrongly in both runs, so the floor turns a wrong answer into an honest refusal.
Correct answers only start being lost at −2.

Two unanswerable questions score **above** the floor (u11 quantization, −1.3; u07
speech-to-text, −4.6) because the corpus discusses neighbouring topics. No score threshold
can catch those; the generator declined both in every run. The two defences cover each
other's blind spots, which is the argument for keeping both.

## In the product

`retrieval/reranker.py` holds `KB_ANSWER_MIN_SCORE` (env-overridable) and
`below_answer_floor()`; `rerank_with_scores()` exposes the scores `rerank()` used to discard.
The orchestrator's knowledge-base path checks the floor between reranking and generation,
skips it when the question carries its own material (an uploaded image, inline numbers), and
marks the response and its trace `abstained`, so `GET /api/traces/summary` reports the
decline rate. Verified live:

| request | route | result |
|---|---|---|
| "in my documents, what is the boiling point of ethanol?" | knowledge base | declined in 0.5 s, no generation |
| "from my uploaded documents, what does Tamim say about … resigning?" | knowledge base | answered from the journal, 4.9 s |
| "what is the boiling point of ethanol?" (no documents cue) | web search | 78.37 °C, with a real source |

The third row is the point of doing this in the knowledge-base path only: the system still
answers general questions, from the web, with a citation that means something.

## Limits

- **The detector is a heuristic.** `is_abstention()` looks for refusal phrasing in the first
  320 characters. Two errors were found by reading outputs and fixed with tests: a missed
  refusal ("No one is explicitly named…") and a false positive ("methods… do not have
  context depth"). Read `results/abstention_*.json` before trusting a changed number.
- **24 questions, one author.** The near set was written by someone who knows the corpus.
- **Run-to-run noise is real.** Generation is sampled and unseeded: the generator alone
  declined 21 of 24 in one run and 22 in the other, and answerable contains-gold moved
  0.734 → 0.703 between two runs of the *identical* generation path. Differences of two or
  three questions are inside that noise; 2-of-4 → 4-of-4 on the far set is not.
- Out-of-corpus questions are not always unanswerable for the *system*: without a
  "my documents" cue the router sends them to the web. This suite tests the
  knowledge-base answer path, deliberately.
