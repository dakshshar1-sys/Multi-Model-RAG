# Evaluation harness

Measures the real retrieval, routing and generation components against labelled data,
so "it works" becomes numbers you can put in a results chapter and compare across changes.

## Layout

    eval/
      corpus/            three project write-ups: the evaluation knowledge base
      datasets/
        routing.jsonl    32 queries with the tool the router must pick
        qa.jsonl         64 questions with gold answer + verbatim gold evidence
      index/             FAISS index built from corpus/ (generated, not committed)
      results/           one JSON per run + history.md (generated, not committed)
      metrics.py         pure metric functions (unit-tested in tests/test_eval_metrics.py)
      build_index.py     builds index/ with the production splitter, embeddings and store
      run_eval.py        the runner

The eval index is separate from the live knowledge base. Runs never touch it.

## Run

    cd backend
    python -m eval.build_index
    python -m eval.run_eval --suite routing --runs 3
    python -m eval.run_eval --suite retrieval
    python -m eval.run_eval --suite retrieval --no-rerank          # ablation
    python -m eval.run_eval --suite answers --limit 10 --judge     # slow
    python -m eval.run_eval --suite all --label baseline

Inside Docker: `docker exec -w /app multimodelrag-backend-1 python -m eval.run_eval ...`

## Metrics

| suite | metric | meaning |
|---|---|---|
| routing | accuracy (mean of N runs), majority accuracy, unstable cases | the classifier is stochastic; report both the mean and how many cases flip |
| retrieval | recall@5, recall@10, MRR, before and after reranking | did the chunk containing the gold evidence come back, and how high |
| answers | token-F1, contains-gold | closeness to the gold answer (content tokens, plural-insensitive) |
| answers | lexical support | fraction of answer sentences whose words are mostly in the context: a model-free faithfulness proxy |
| answers | judge PASS rate (`--judge`) | the existing verifier as an LLM judge; disagreements with the proxy are listed for reading by hand |
| answers | contains-gold given evidence | conditional on retrieval having succeeded: isolates generation quality from retrieval quality |

Gold evidence is a short verbatim substring of the source document. A retrieval "hit"
means a returned chunk contains it (case/whitespace-insensitive). This survives changes to
chunk size and does not depend on page numbers.

## Labelling notes

Every item in `qa.jsonl` starts with `"reviewed": false`. Read the question, gold answer and
evidence against the corpus and flip it to `true`. Gold answers reflect what the *document*
says, even where the document and the code disagree (the architecture write-up names a
BGE reranker and Llama 3.2; the code uses ms-marco-MiniLM and Qwen). The harness measures
grounding in the corpus, not the truth of the corpus.

Add items by appending a line. Keep evidence under ~120 characters so it sits inside one chunk.

## Routing: strict vs effective

`expected` is the tool the router itself should choose. An optional `accept` list names
other tools that still produce the right outcome because a deterministic guard in the
orchestrator redirects them (e.g. `Visualize_Data` with no numbers in the message is sent
to `Web_Search`). The runner reports **strict** accuracy (router alone) and **effective**
accuracy (router plus guards). The gap between them is the value of the guards.
