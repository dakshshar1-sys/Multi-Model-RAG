"""
Evaluation harness: measures the real components against labelled datasets.

    cd backend
    python -m eval.build_index                      # once, or after editing eval/corpus
    python -m eval.run_eval --suite routing --runs 3
    python -m eval.run_eval --suite retrieval
    python -m eval.run_eval --suite retrieval --no-rerank        # ablation
    python -m eval.run_eval --suite answers --limit 10 --judge   # slow: generation
    python -m eval.run_eval --suite all --label baseline

Writes eval/results/<label>_<timestamp>.json (config, per-item rows, aggregates)
and appends one line to eval/results/history.md so ablations can be compared.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (contains_gold, lexical_support, mean, mrr, recall_at_k, token_f1,
                          evidence_rank)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "datasets")
RESULTS = os.path.join(HERE, "results")
INDEX = os.path.join(HERE, "index")
logger = logging.getLogger("eval")


def load_jsonl(name: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


# ── routing ────────────────────────────────────────────────────────────────

def eval_routing(runs: int, model_choice: str) -> dict:
    from models.agentic_router import AgentRouter
    router = AgentRouter()
    cases = load_jsonl("routing.jsonl")
    rows, per_run_acc, per_run_eff = [], [], []
    for r in range(runs):
        correct = eff_correct = 0
        for c in cases:
            t0 = time.perf_counter()
            got = router.route_query(c["query"], model_choice, c["query"])
            dt = time.perf_counter() - t0
            ok = got == c["expected"]
            effective = ok or got in c.get("accept", [])
            correct += ok
            eff_correct += effective
            rows.append({"run": r, "id": c["id"], "tag": c.get("tag"), "query": c["query"],
                         "expected": c["expected"], "got": got, "correct": ok, "effective": effective,
                         "latency_s": round(dt, 3)})
        per_run_acc.append(correct / len(cases))
        per_run_eff.append(eff_correct / len(cases))
    # per-case majority vote across runs, and a confusion table
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["id"]].append(row["got"])
    majority_correct = 0
    confusion = Counter()
    for c in cases:
        votes = Counter(by_case[c["id"]])
        top = votes.most_common(1)[0][0]
        majority_correct += top == c["expected"]
        for got in by_case[c["id"]]:
            if got != c["expected"]:
                confusion[(c["expected"], got)] += 1
    unstable = [cid for cid, gots in by_case.items() if len(set(gots)) > 1]
    return {
        "n_cases": len(cases), "runs": runs,
        "accuracy_mean": round(mean(per_run_acc), 4), "accuracy_per_run": [round(a, 4) for a in per_run_acc],
        "effective_accuracy_mean": round(mean(per_run_eff), 4),
        "majority_accuracy": round(majority_correct / len(cases), 4),
        "unstable_cases": unstable,
        "confusion": [{"expected": e, "got": g, "count": n} for (e, g), n in confusion.most_common()],
        "latency_mean_s": round(mean([r["latency_s"] for r in rows]), 3),
        "rows": rows,
    }


# ── retrieval ──────────────────────────────────────────────────────────────

def _eval_db():
    from retrieval.vector_db import VectorDatabase
    if not (os.path.isdir(INDEX) and os.listdir(INDEX)):
        from eval.build_index import build
        logger.info("No eval index yet; building it.")
        return build()
    return VectorDatabase(index_path=INDEX)


def _retriever(db, mode: str):
    return {"dense": db.retrieve, "bm25": db.bm25_retrieve, "hybrid": db.hybrid_retrieve}[mode]


def eval_retrieval(top_k_initial: int, top_k_final: int, use_rerank: bool, limit: int | None, mode: str = "dense", dataset: str = "qa") -> dict:
    db = _eval_db()
    reranker = None
    if use_rerank:
        from retrieval.reranker import RerankerModel
        reranker = RerankerModel()
    qa = load_jsonl(f"{dataset}.jsonl", limit)
    retrieve = _retriever(db, mode)
    rows = []
    for q in qa:
        t0 = time.perf_counter()
        docs = retrieve(q["question"], top_k=top_k_initial)
        texts = [d.page_content for d in docs]
        t_ret = time.perf_counter() - t0
        pre_rank = evidence_rank(q["gold_evidence"], texts)
        row = {"id": q["id"], "source": q["source"], "question": q["question"],
               "pre_rank": pre_rank,
               f"pre_recall@{top_k_final}": recall_at_k(q["gold_evidence"], texts, top_k_final),
               f"pre_recall@{top_k_initial}": recall_at_k(q["gold_evidence"], texts, top_k_initial),
               "pre_mrr": mrr(q["gold_evidence"], texts), "retrieve_s": round(t_ret, 3)}
        if reranker:
            t1 = time.perf_counter()
            ranked = reranker.rerank(q["question"], texts, top_k=top_k_final)
            row["rerank_s"] = round(time.perf_counter() - t1, 3)
            row["post_rank"] = evidence_rank(q["gold_evidence"], ranked)
            row[f"post_recall@{top_k_final}"] = recall_at_k(q["gold_evidence"], ranked, top_k_final)
            row["post_mrr"] = mrr(q["gold_evidence"], ranked)
        rows.append(row)
    agg = {"n": len(rows), "top_k_initial": top_k_initial, "top_k_final": top_k_final, "rerank": use_rerank, "mode": mode,
           f"pre_recall@{top_k_final}": round(mean([r[f"pre_recall@{top_k_final}"] for r in rows]), 4),
           f"pre_recall@{top_k_initial}": round(mean([r[f"pre_recall@{top_k_initial}"] for r in rows]), 4),
           "pre_mrr": round(mean([r["pre_mrr"] for r in rows]), 4),
           "retrieve_mean_s": round(mean([r["retrieve_s"] for r in rows]), 3)}
    if reranker:
        agg[f"post_recall@{top_k_final}"] = round(mean([r[f"post_recall@{top_k_final}"] for r in rows]), 4)
        agg["post_mrr"] = round(mean([r["post_mrr"] for r in rows]), 4)
        agg["rerank_mean_s"] = round(mean([r["rerank_s"] for r in rows]), 3)
    agg["misses"] = [r["id"] for r in rows if r["pre_rank"] is None]
    agg["by_source"] = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        agg["by_source"][src] = {"n": len(sub), f"pre_recall@{top_k_final}": round(mean([r[f"pre_recall@{top_k_final}"] for r in sub]), 4)}
    agg["rows"] = rows
    return agg


# ── answers ────────────────────────────────────────────────────────────────

async def eval_answers(top_k_initial: int, top_k_final: int, use_rerank: bool, judge: bool,
                       limit: int | None, model_choice: str, mode: str = "dense", dataset: str = "qa") -> dict:
    from models.generation import GenerationModel
    db = _eval_db()
    reranker = None
    if use_rerank:
        from retrieval.reranker import RerankerModel
        reranker = RerankerModel()
    gen = GenerationModel()
    verifier = None
    if judge:
        from verification.verifier import VerificationModule
        verifier = VerificationModule()
    qa = load_jsonl(f"{dataset}.jsonl", limit)
    retrieve = _retriever(db, mode)
    rows = []
    for q in qa:
        docs = retrieve(q["question"], top_k=top_k_initial)
        texts = [d.page_content for d in docs]
        srcs = [d.metadata.get("source", "?") for d in docs]
        context = reranker.rerank(q["question"], texts, top_k=top_k_final) if reranker else texts[:top_k_final]
        t0 = time.perf_counter()
        answer = await gen.generate_answer(q["question"], context, sources=srcs[:len(context)],
                                           mode="analytical", model_choice=model_choice)
        t_gen = time.perf_counter() - t0
        row = {"id": q["id"], "question": q["question"], "gold": q["gold_answer"], "answer": answer,
               "evidence_in_context": evidence_rank(q["gold_evidence"], context) is not None,
               "token_f1": round(token_f1(answer, q["gold_answer"]), 4),
               "contains_gold": contains_gold(answer, q["gold_answer"]),
               "lexical_support": round(lexical_support(answer, context), 4),
               "answer_chars": len(answer), "generate_s": round(t_gen, 2)}
        if verifier:
            ok, reason = await verifier.verify(answer, context, model_choice=model_choice)
            row["judge_pass"] = bool(ok)
            row["judge_reason"] = reason
        rows.append(row)
        logger.info(f"{q['id']} f1={row['token_f1']} contains={row['contains_gold']} support={row['lexical_support']}")
    agg = {"n": len(rows), "rerank": use_rerank, "judge": judge, "mode": mode,
           "evidence_in_context_rate": round(mean([float(r["evidence_in_context"]) for r in rows]), 4),
           "token_f1_mean": round(mean([r["token_f1"] for r in rows]), 4),
           "contains_gold_rate": round(mean([r["contains_gold"] for r in rows]), 4),
           "lexical_support_mean": round(mean([r["lexical_support"] for r in rows]), 4),
           "generate_mean_s": round(mean([r["generate_s"] for r in rows]), 2)}
    if verifier:
        agg["judge_pass_rate"] = round(mean([float(r["judge_pass"]) for r in rows]), 4)
        # where the cheap proxy and the LLM judge disagree is what to read by hand
        agg["proxy_vs_judge_disagreements"] = [r["id"] for r in rows
                                               if (r["lexical_support"] >= 0.6) != r["judge_pass"]]
    # conditional: when the evidence WAS in context, how often did the answer contain the gold?
    with_ev = [r for r in rows if r["evidence_in_context"]]
    agg["contains_gold_given_evidence"] = round(mean([r["contains_gold"] for r in with_ev]), 4) if with_ev else None
    agg["rows"] = rows
    return agg


# ── reporting ──────────────────────────────────────────────────────────────

def summarize(results: dict) -> str:
    lines = []
    if "routing" in results:
        r = results["routing"]
        lines.append(f"| routing | strict accuracy {r['accuracy_mean']:.3f} (mean of {r['runs']} runs) | effective {r['effective_accuracy_mean']:.3f} | majority {r['majority_accuracy']:.3f} | unstable {len(r['unstable_cases'])}/{r['n_cases']} | {r['latency_mean_s']}s/case |")
    if "retrieval" in results:
        r = results["retrieval"]; k, K = r["top_k_final"], r["top_k_initial"]
        post = f" | post-rerank recall@{k} {r[f'post_recall@{k}']:.3f}, MRR {r['post_mrr']:.3f}" if r["rerank"] else " | rerank OFF"
        lines.append(f"| retrieval ({r.get('mode','dense')}) | n={r['n']} | recall@{k} {r[f'pre_recall@{k}']:.3f}, recall@{K} {r[f'pre_recall@{K}']:.3f}, MRR {r['pre_mrr']:.3f}{post} | misses {len(r['misses'])} |")
    if "answers" in results:
        r = results["answers"]
        judge = f" | judge PASS {r['judge_pass_rate']:.3f}" if r.get("judge") else ""
        lines.append(f"| answers ({r.get('mode','dense')}) | n={r['n']} | token-F1 {r['token_f1_mean']:.3f} | contains-gold {r['contains_gold_rate']:.3f} (given evidence {r['contains_gold_given_evidence']}) | lexical support {r['lexical_support_mean']:.3f}{judge} | {r['generate_mean_s']}s/answer |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["routing", "retrieval", "answers", "all"], default="all")
    ap.add_argument("--runs", type=int, default=3, help="routing: repeat runs (the classifier is stochastic)")
    ap.add_argument("--top-k", type=int, default=5, help="final top-k passed to the LLM")
    ap.add_argument("--top-k-initial", type=int, default=10, help="FAISS candidates before reranking")
    ap.add_argument("--no-rerank", action="store_true", help="ablation: skip the cross-encoder")
    ap.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default="dense", help="retrieval mode (ablation)")
    ap.add_argument("--judge", action="store_true", help="answers: also run the LLM verifier as a judge")
    ap.add_argument("--limit", type=int, default=None, help="only the first N qa items (quick runs)")
    ap.add_argument("--model", default="auto", help="model_choice for LLM calls: auto|local|api|claude")
    ap.add_argument("--label", default="run", help="name for the results file / history line")
    ap.add_argument("--dataset", default="qa", help="qa (author-written) or qa_independent (written from headings by people who have not read the corpus)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    results = {"config": vars(args), "started": datetime.now().isoformat(timespec="seconds")}
    t0 = time.time()
    if args.suite in ("routing", "all"):
        results["routing"] = eval_routing(args.runs, args.model)
    if args.suite in ("retrieval", "all"):
        results["retrieval"] = eval_retrieval(args.top_k_initial, args.top_k, not args.no_rerank, args.limit, args.retrieval, args.dataset)
    if args.suite in ("answers", "all"):
        results["answers"] = asyncio.run(eval_answers(args.top_k_initial, args.top_k, not args.no_rerank,
                                                      args.judge, args.limit, args.model, args.retrieval, args.dataset))
    results["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS, f"{args.label}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary = summarize(results)
    with open(os.path.join(RESULTS, "history.md"), "a", encoding="utf-8") as f:
        f.write(f"\n### {args.label} — {results['started']} ({results['elapsed_s']}s)\n\n{summary}\n")
    print("\n" + summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
