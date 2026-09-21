"""
Whole-pipeline ablation: what is each component worth, in answer correctness?

    cd backend
    python -m eval.ablation                      # five rows, all questions (~30 min)
    python -m eval.ablation --limit 5            # smoke
    python -m eval.ablation --e2e                # add the real orchestrator row (routing, verification)
    python -m eval.ablation --dataset qa_independent --label indep

Rows (each adds one component to the previous):
  no_retrieval    the model alone, no context          (is retrieval worth anything?)
  dense           FAISS top-5                           (baseline RAG)
  dense+rerank    FAISS top-10 -> cross-encoder top-5   (+ reranker)
  hybrid          BM25+dense fused, top-5               (+ lexical retrieval)
  hybrid+rerank   fused top-10 -> cross-encoder top-5   (+ both: the shipped retrieval stack)
  full_e2e        the real orchestrator in-process: routing, hybrid+rerank, generation,
                  verification; cache disabled; index swapped to the eval corpus  (+ routing)

Metrics per row: contains-gold, token-F1, lexical support, evidence-in-context, seconds.
Writes eval/results/ablation_<label>_<stamp>.json and appends the table to history.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import contains_gold, evidence_rank, lexical_support, mean, token_f1
from eval.run_eval import RESULTS, _eval_db, load_jsonl

logger = logging.getLogger("ablation")

LADDER = [
    # name,            retrieval mode, top_k_initial, rerank, top_k_final
    ("no_retrieval",   None,    0,  False, 0),
    ("dense",          "dense", 5,  False, 5),
    ("dense+rerank",   "dense", 10, True,  5),
    ("hybrid",         "hybrid", 5, False, 5),
    ("hybrid+rerank",  "hybrid", 10, True, 5),
]


def _score(q: dict, answer: str, context: list[str], seconds: float) -> dict:
    return {"id": q["id"], "answer": answer[:600], "seconds": round(seconds, 2),
            "contains_gold": contains_gold(answer, q["gold_answer"]),
            "token_f1": round(token_f1(answer, q["gold_answer"]), 4),
            "lexical_support": round(lexical_support(answer, context), 4) if context else 0.0,
            "evidence_in_context": bool(context) and evidence_rank(q["gold_evidence"], context) is not None}


def _aggregate(name: str, rows: list[dict]) -> dict:
    return {"row": name, "n": len(rows),
            "contains_gold": round(mean([r["contains_gold"] for r in rows]), 3),
            "token_f1": round(mean([r["token_f1"] for r in rows]), 3),
            "lexical_support": round(mean([r["lexical_support"] for r in rows]), 3),
            "evidence_in_context": round(mean([float(r["evidence_in_context"]) for r in rows]), 3),
            "seconds": round(mean([r["seconds"] for r in rows]), 2)}


async def run_ladder(qa: list[dict], model_choice: str, rows_wanted: list[str]) -> tuple[list[dict], dict]:
    from models.generation import GenerationModel
    from retrieval.reranker import RerankerModel
    db = _eval_db()
    gen = GenerationModel()
    reranker = RerankerModel()
    aggregates, per_item = [], {}
    for name, mode, k_init, rerank, k_final in LADDER:
        if name not in rows_wanted:
            continue
        rows = []
        for q in qa:
            t0 = time.perf_counter()
            if mode is None:
                context = []
                answer = await gen.generate_answer(q["question"], mode="conversational", model_choice=model_choice)
            else:
                retrieve = {"dense": db.retrieve, "hybrid": db.hybrid_retrieve}[mode]
                docs = retrieve(q["question"], top_k=k_init)
                texts = [d.page_content for d in docs]
                srcs = [d.metadata.get("source", "?") for d in docs]
                context = reranker.rerank(q["question"], texts, top_k=k_final) if rerank else texts[:k_final]
                answer = await gen.generate_answer(q["question"], context, sources=srcs[:len(context)],
                                                   mode="analytical", model_choice=model_choice)
            rows.append(_score(q, answer, context, time.perf_counter() - t0))
            logger.info(f"[{name}] {q['id']} contains={rows[-1]['contains_gold']} f1={rows[-1]['token_f1']}")
        aggregates.append(_aggregate(name, rows)); per_item[name] = rows
    return aggregates, per_item


async def run_e2e(qa: list[dict], model_choice: str) -> tuple[dict, list[dict]]:
    """The real orchestrator, in-process, against the eval index, cache off."""
    from orchestrator.master_llm import MasterOrchestrator
    orch = MasterOrchestrator()
    orch.vector_db = _eval_db()
    orch.cache.get = lambda *a, **k: None          # never serve a cached answer
    orch.cache.set = lambda *a, **k: None          # never poison the real cache
    rows, tools = [], {}
    for q in qa:
        t0 = time.perf_counter()
        answer, tool, context = "", None, []
        async for raw in orch.process_query_stream(q["question"], "", "", model_choice, conversation_id="ablation"):
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            a = ev.get("action", "")
            if ev.get("model") == "Agent Router" and a.startswith("Selected Tool: ["):
                tool = a.split("[")[-1].rstrip("]")
            d = ev.get("details") or {}
            if ev.get("model") == "Final Response" and d.get("answer"):
                answer = d["answer"]
        # context for lexical support: whatever the eval index would have given (best effort)
        docs = orch.vector_db.hybrid_retrieve(q["question"], top_k=5)
        context = [d.page_content for d in docs]
        row = _score(q, answer, context, time.perf_counter() - t0); row["tool"] = tool
        rows.append(row); tools[tool] = tools.get(tool, 0) + 1
        logger.info(f"[full_e2e] {q['id']} tool={tool} contains={row['contains_gold']}")
    agg = _aggregate("full_e2e", rows); agg["tools"] = tools
    return agg, rows


def format_table(aggregates: list[dict]) -> str:
    lines = ["| configuration | contains-gold | token-F1 | lexical support | evidence in context | s/answer |",
             "|---|---|---|---|---|---|"]
    for a in aggregates:
        extra = f" (tools: {a['tools']})" if a.get("tools") else ""
        lines.append(f"| {a['row']}{extra} | {a['contains_gold']:.3f} | {a['token_f1']:.3f} | {a['lexical_support']:.3f} | {a['evidence_in_context']:.3f} | {a['seconds']:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="qa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--rows", default=",".join(n for n, *_ in LADDER), help="comma-separated subset of ladder rows")
    ap.add_argument("--e2e", action="store_true", help="also run the real orchestrator row (slow)")
    ap.add_argument("--label", default="ablation")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "faiss", "urllib3", "retrieval", "models", "core", "verification", "orchestrator"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    qa = load_jsonl(f"{args.dataset}.jsonl", args.limit)
    t0 = time.time()
    aggregates, per_item = asyncio.run(run_ladder(qa, args.model, args.rows.split(",")))
    if args.e2e:
        agg, rows = asyncio.run(run_e2e(qa, args.model))
        aggregates.append(agg); per_item["full_e2e"] = rows
    table = format_table(aggregates)
    out = {"config": vars(args), "started": datetime.now().isoformat(timespec="seconds"),
           "elapsed_s": round(time.time() - t0, 1), "n": len(qa), "aggregates": aggregates, "per_item": per_item}
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"ablation_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS, "history.md"), "a", encoding="utf-8") as f:
        f.write(f"\n### ablation {args.label} ({args.dataset}, n={len(qa)}) — {out['started']} ({out['elapsed_s']}s)\n\n{table}\n")
    print("\n" + table + f"\n\nwrote {path}")


if __name__ == "__main__":
    main()
