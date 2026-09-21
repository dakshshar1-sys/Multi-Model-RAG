"""
Abstention evaluation: when the documents do not contain the answer, does the system say so?

    cd backend
    python -m eval.abstention                  # both sets, ~8 min
    python -m eval.abstention --limit 6        # smoke

Every other suite asks answerable questions, so a system that always answers looks perfect
on them. This one runs datasets/unanswerable.jsonl (24 questions whose facts are verified
absent from the corpus) through the shipped knowledge-base path — hybrid top-10, cross-encoder
top-5, analytical generation — and reports how often the answer declines instead of inventing.
The 64 answerable questions run through the same path to price the other side: how often the
system declines when it should have answered.

Two defences are measured separately, because they fail differently:
  generator   the model, given weak context, says the documents do not contain it
  gate        the best cross-encoder score is below a threshold, so nothing retrieved is
              about the question and no generation is attempted
The sweep table shows, per threshold, unanswerable questions the gate would stop and answerable
ones it would wrongly stop — split by whether their answer was wrong anyway.
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

from core.text_support import is_abstention
from eval.metrics import contains_gold, mean
from eval.run_eval import RESULTS, _eval_db, load_jsonl

logger = logging.getLogger("abstention")
SWEEP = (-8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0)


def sweep(unanswerable: list[dict], answerable: list[dict], thresholds=SWEEP) -> list[dict]:
    """Offline gate analysis over recorded best scores. A None score (reranker off) never gates."""
    def gated(r, t):
        return r["best_score"] is not None and r["best_score"] < t
    out = []
    for t in thresholds:
        u_gate = [r for r in unanswerable if gated(r, t)]
        u_total = [r for r in unanswerable if gated(r, t) or r["abstained"]]
        a_gate = [r for r in answerable if gated(r, t)]
        out.append({"threshold": t,
                    "unanswerable_stopped_by_gate": len(u_gate),
                    "unanswerable_declined_total": len(u_total),
                    "unanswerable_n": len(unanswerable),
                    "answerable_wrongly_gated": len(a_gate),
                    "of_which_were_correct": sum(1 for r in a_gate if r.get("contains_gold")),
                    "answerable_n": len(answerable)})
    return out


def summarize(unanswerable: list[dict], answerable: list[dict], gate: float | None) -> dict:
    def declined(r):
        return r["abstained"] or (gate is not None and r["best_score"] is not None and r["best_score"] < gate)
    u_decl = [r for r in unanswerable if declined(r)]
    a_decl = [r for r in answerable if declined(r) and not r.get("contains_gold")]
    by_kind = {}
    for kind in sorted({r["kind"] for r in unanswerable}):
        rows = [r for r in unanswerable if r["kind"] == kind]
        by_kind[kind] = {"n": len(rows), "declined": sum(1 for r in rows if declined(r))}
    return {"gate": gate,
            "unanswerable": {"n": len(unanswerable), "declined": len(u_decl),
                             "abstention_rate": round(len(u_decl) / max(1, len(unanswerable)), 3),
                             "by_generator_alone": sum(1 for r in unanswerable if r["abstained"]),
                             "by_kind": by_kind,
                             "invented": [r["id"] for r in unanswerable if not declined(r)]},
            "answerable": {"n": len(answerable),
                           "false_abstentions": len(a_decl),
                           "false_abstention_rate": round(len(a_decl) / max(1, len(answerable)), 3),
                           "contains_gold": round(mean([float(r["contains_gold"]) for r in answerable]), 3) if answerable else None,
                           "false_abstention_ids": [r["id"] for r in a_decl]},
            "best_score": {"unanswerable_median": _median([r["best_score"] for r in unanswerable]),
                           "answerable_median": _median([r["best_score"] for r in answerable])}}


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 2)


async def run_set(rows: list[dict], gen, db, reranker, model_choice: str, answerable: bool) -> list[dict]:
    out = []
    for q in rows:
        t0 = time.perf_counter()
        docs = db.hybrid_retrieve(q["question"], top_k=10)
        texts = [d.page_content for d in docs]
        srcs = [d.metadata.get("source", "?") for d in docs]
        scored = reranker.rerank_with_scores(q["question"], texts, top_k=5)
        context = [doc for _, doc in scored]
        best = scored[0][0] if scored else None
        answer = await gen.generate_answer(q["question"], context, sources=srcs[:len(context)], mode="analytical", model_choice=model_choice)
        row = {"id": q["id"], "kind": q.get("kind", "answerable"), "best_score": None if best is None else round(best, 3),
               "abstained": is_abstention(answer), "answer": answer[:500], "seconds": round(time.perf_counter() - t0, 2)}
        if answerable:
            row["contains_gold"] = bool(contains_gold(answer, q["gold_answer"]))
        out.append(row)
        logger.info(f"{q['id']} best={row['best_score']} abstained={row['abstained']}" + (f" gold={row['contains_gold']}" if answerable else ""))
    return out


def format_tables(summary: dict, sw: list[dict]) -> str:
    u, a = summary["unanswerable"], summary["answerable"]
    gate = "off" if summary["gate"] is None else f"{summary['gate']:g}"
    lines = [f"| set | n | declined | rate | notes |", "|---|---|---|---|---|",
             f"| unanswerable (should decline) | {u['n']} | {u['declined']} | **{u['abstention_rate']:.3f}** | generator alone {u['by_generator_alone']}; "
             + "; ".join(f"{k} {v['declined']}/{v['n']}" for k, v in u["by_kind"].items()) + f"; gate {gate} |",
             f"| answerable (should answer) | {a['n']} | {a['false_abstentions']} | {a['false_abstention_rate']:.3f} | false abstentions; contains-gold {a['contains_gold']} |",
             "", f"best cross-encoder score, median: unanswerable {summary['best_score']['unanswerable_median']}, answerable {summary['best_score']['answerable_median']}", "",
             "| gate threshold | unanswerable stopped by gate | unanswerable declined (gate or generator) | answerable wrongly gated | …of which had been correct |", "|---|---|---|---|---|"]
    for r in sw:
        lines.append(f"| {r['threshold']:g} | {r['unanswerable_stopped_by_gate']}/{r['unanswerable_n']} | {r['unanswerable_declined_total']}/{r['unanswerable_n']} | {r['answerable_wrongly_gated']}/{r['answerable_n']} | {r['of_which_were_correct']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--gate", default=None, help="best-score floor below which the system declines; default = production's KB_ANSWER_MIN_SCORE; 'none' = generator only")
    ap.add_argument("--skip-answerable", action="store_true")
    ap.add_argument("--label", default="abstention")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "sentence_transformers", "faiss", "urllib3", "retrieval", "models", "core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from retrieval.reranker import KB_ANSWER_MIN_SCORE
    gate = KB_ANSWER_MIN_SCORE if args.gate is None else (None if str(args.gate).lower() == "none" else float(args.gate))

    from models.generation import GenerationModel
    from retrieval.reranker import RerankerModel
    db, gen, reranker = _eval_db(), GenerationModel(), RerankerModel()
    t0 = time.time()
    un = asyncio.run(run_set(load_jsonl("unanswerable.jsonl", args.limit), gen, db, reranker, args.model, answerable=False))
    an = [] if args.skip_answerable else asyncio.run(run_set(load_jsonl("qa.jsonl", args.limit), gen, db, reranker, args.model, answerable=True))
    summary, sw = summarize(un, an, gate), sweep(un, an)
    table = format_tables(summary, sw)
    out = {"config": vars(args), "started": datetime.now().isoformat(timespec="seconds"), "elapsed_s": round(time.time() - t0, 1),
           "summary": summary, "sweep": sw, "unanswerable": un, "answerable": an}
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"abstention_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS, "history.md"), "a", encoding="utf-8") as f:
        f.write(f"\n### abstention {args.label} — {out['started']} ({out['elapsed_s']}s)\n\n{table}\n")
    print("\n" + table + f"\n\nwrote {path}")


if __name__ == "__main__":
    main()
