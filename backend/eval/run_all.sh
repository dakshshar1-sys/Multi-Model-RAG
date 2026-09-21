#!/usr/bin/env sh
# One command reproduces every number in eval/BASELINE.md.
#
#   cd backend && sh eval/run_all.sh                 # local model from LLM_MODEL (default in .env / compose)
#   cd backend && LLM_MODEL=llama3.2 sh eval/run_all.sh llama32   # same suites, another model, labelled
#   Inside Docker: docker exec -w /app multimodelrag-backend-1 sh eval/run_all.sh
#
# Takes ~50 minutes on a 4 GB GPU with a 3B model (the answers suite and the ablation
# dominate). Results land in eval/results/ and one line per run in results/history.md.
set -e
LABEL="${1:-repro}"
cd "$(dirname "$0")/.."
echo "== index ==";        python -B -W ignore -m eval.build_index
echo "== routing ==";      python -B -W ignore -m eval.run_eval --suite routing --runs 3 --label "routing_$LABEL" | grep -E '^\|'
echo "== retrieval ==";    for m in dense hybrid; do python -B -W ignore -m eval.run_eval --suite retrieval --retrieval $m --label "ret_${m}_$LABEL" | grep -E '^\|'; done
                           python -B -W ignore -m eval.run_eval --suite retrieval --retrieval bm25 --no-rerank --label "ret_bm25_$LABEL" | grep -E '^\|'
echo "== answers ==";      python -B -W ignore -m eval.run_eval --suite answers --judge --retrieval hybrid --label "answers_$LABEL" | grep -E '^\|'
echo "== ablation ==";     python -B -W ignore -m eval.ablation --label "$LABEL" | grep -E '^\|'
echo "== appendix ==";     python -B -W ignore -m eval.failure_analysis
echo "== latency ==";      python -B -W ignore -m eval.latency_report
echo "done: see eval/results/history.md and eval/FAILURE_ANALYSIS.md"
