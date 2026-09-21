# Independent question set — protocol

## Why
Every question in `qa.jsonl` was written by reading the documents, so the questions reuse
the documents' own vocabulary. That favours lexical matching: on that set BM25 alone
scored highest (recall@5 0.969), above hybrid (0.938) and dense (0.781). Real users
paraphrase. To learn whether hybrid beats lexical on questions people actually ask, the
questions must come from people who have not read the text.

## Procedure
1. Run `python -m eval.make_heading_sheet` and give `HEADINGS_SHEET.md` to **two people
   who have not read the corpus** (classmates are fine). They write ~20 questions each
   from the section titles only. No answers, no peeking at the text.
2. For each returned question, the author (you) finds the answer in the corpus and fills
   one line in `qa_independent.jsonl`:
   `{"id": "i01", "source": "<file>", "question": "<verbatim>", "gold_answer": "...",
     "gold_evidence": "<short verbatim substring from the document>", "author": "<initials>",
     "reviewed": true}`
   Keep `gold_evidence` under ~120 characters so it sits inside one chunk. If a question
   has no answer in the corpus, keep it with `"gold_answer": null` — unanswerable
   questions are a useful separate count (the system should say it does not know).
3. Run both sets with identical settings and compare:
   ```
   python -m eval.run_eval --suite retrieval --retrieval dense  --dataset qa_independent --label indep_dense
   python -m eval.run_eval --suite retrieval --retrieval bm25 --no-rerank --dataset qa_independent --label indep_bm25
   python -m eval.run_eval --suite retrieval --retrieval hybrid --dataset qa_independent --label indep_hybrid
   python -m eval.ablation --dataset qa_independent --label indep
   ```
4. Report the two sets side by side. The expected pattern is that lexical retrieval loses
   ground on paraphrased questions and hybrid holds. If that happens you have a finding
   about question style; if it does not, you have a different finding. Both are results.

## What not to do
- Do not rewrite the classmates' questions to make them "clearer": the paraphrase is the point.
- Do not drop questions the system gets wrong.
- Do not let the question-writers see `qa.jsonl`.
