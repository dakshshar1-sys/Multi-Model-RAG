"""
Failure-analysis appendix, generated from the harness's own result files.

    cd backend && python -m eval.failure_analysis        # -> eval/FAILURE_ANALYSIS.md

Sections: retrieval misses (with dense vs hybrid ranks), generation faults (the
evidence was in context yet the gold fact is missing from the answer), judge vs
lexical-proxy disagreements, routing confusions, and what the model answers with
no retrieval at all. Every row carries the question, the gold, and the system's
output, so a reader can see the failure rather than take a number on trust.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "FAILURE_ANALYSIS.md")


def latest(prefix):
    from eval.results_io import latest_result
    return latest_result(prefix, RESULTS)


def _q(text: str, n: int = 110) -> str:
    text = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def build(qa: dict[str, dict], answers: dict | None, ret_dense: dict | None, ret_hybrid: dict | None,
          routing: dict | None, ablation: dict | None) -> str:
    out = ["# Failure analysis (generated)\n",
           "Produced by `python -m eval.failure_analysis` from the latest result files. Rows are the",
           "system's actual outputs; nothing here is edited by hand.\n"]

    # 1. retrieval misses
    if ret_dense:
        rows_d = {r["id"]: r for r in ret_dense["retrieval"]["rows"]}
        rows_h = {r["id"]: r for r in ret_hybrid["retrieval"]["rows"]} if ret_hybrid else {}
        k = ret_dense["retrieval"]["top_k_final"]
        misses = [i for i, r in rows_d.items() if not r.get(f"pre_recall@{k}")]
        out.append(f"\n## 1. Retrieval misses (dense top-{k}) — {len(misses)} of {len(rows_d)}\n")
        out.append("| id | question | source | gold evidence | dense rank | hybrid rank |")
        out.append("|---|---|---|---|---|---|")
        for i in misses:
            q = qa.get(i, {}); h = rows_h.get(i, {})
            out.append(f"| {i} | {_q(q.get('question'), 80)} | {q.get('source','')} | {_q(q.get('gold_evidence'), 60)} | {rows_d[i].get('pre_rank')} | {h.get('pre_rank', '–')} |")
        out.append("\nA missing rank means the evidence was not in the top-10 at all. Where the hybrid rank is")
        out.append("small and the dense rank large, the question's wording matched the chunk lexically but not")
        out.append("semantically: typically a short bulleted line whose embedding is dominated by its neighbours.")

    # 2. generation faults
    if answers:
        rows = answers["answers"]["rows"]
        faults = [r for r in rows if r["evidence_in_context"] and not r["contains_gold"]]
        out.append(f"\n## 2. Generation faults — evidence retrieved, gold fact missing from the answer: {len(faults)} of {len(rows)}\n")
        out.append("| id | question | gold answer | answer (excerpt) | lexical support | judge |")
        out.append("|---|---|---|---|---|---|")
        for r in faults:
            q = qa.get(r["id"], {})
            out.append(f"| {r['id']} | {_q(q.get('question'), 70)} | {_q(r['gold'], 70)} | {_q(r['answer'], 120)} | {r['lexical_support']:.2f} | {'PASS' if r.get('judge_pass') else 'FAIL' if 'judge_pass' in r else '–'} |")
        out.append("\nThese are the cases the retrieval stack cannot fix: the fact was in front of the model.")
        out.append("Read the excerpts: the common patterns are paraphrase that drops the key term, a partial")
        out.append("list, or the model answering a neighbouring question from the same chunk.")

        # 3. judge vs proxy
        dis = answers["answers"].get("proxy_vs_judge_disagreements") or []
        by_id = {r["id"]: r for r in rows}
        out.append(f"\n## 3. LLM judge vs lexical proxy disagreements — {len(dis)}\n")
        out.append("| id | lexical support | judge | judge reason | contains gold |")
        out.append("|---|---|---|---|---|")
        for i in dis:
            r = by_id[i]
            out.append(f"| {i} | {r['lexical_support']:.2f} | {'PASS' if r.get('judge_pass') else 'FAIL'} | {_q(r.get('judge_reason'), 90)} | {int(r['contains_gold'])} |")
        out.append("\nA PASS with low support and no gold fact is the judge being lenient; a FAIL with high")
        out.append("support and the gold fact present is the judge being wrong. Count both before trusting it.")

    # 4. routing confusions
    if routing:
        r = routing["routing"]
        out.append(f"\n## 4. Routing confusions — strict accuracy {r['accuracy_mean']:.3f}, unstable {len(r['unstable_cases'])}\n")
        out.append("| expected | got | count | example query |")
        out.append("|---|---|---|---|")
        seen = set()
        for c in r["confusion"]:
            ex = next((x["query"] for x in r["rows"] if x["expected"] == c["expected"] and x["got"] == c["got"]), "")
            out.append(f"| {c['expected']} | {c['got']} | {c['count']} | {_q(ex, 80)} |")
        if r["unstable_cases"]:
            out.append(f"\nUnstable across runs: {', '.join(r['unstable_cases'])} — the sampled classifier gave different answers to the same query.")

    # 5. what the model says with no retrieval
    if ablation and "no_retrieval" in ablation.get("per_item", {}):
        rows = ablation["per_item"]["no_retrieval"]
        right = [r for r in rows if r["contains_gold"]]
        out.append(f"\n## 5. Model alone, no retrieval — {len(right)} of {len(rows)} contain the gold fact\n")
        out.append("| id | question | model's answer (excerpt) | contains gold |")
        out.append("|---|---|---|---|")
        for r in rows[:12]:
            q = qa.get(r["id"], {})
            out.append(f"| {r['id']} | {_q(q.get('question'), 70)} | {_q(r['answer'], 110)} | {int(r['contains_gold'])} |")
        out.append(f"\n(first 12 of {len(rows)} shown; the ones it gets right are general knowledge, e.g. {', '.join(x['id'] for x in right) or 'none'}.)")
    return "\n".join(out) + "\n"


def main():
    qa = {}
    with open(os.path.join(HERE, "datasets", "qa.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                r = json.loads(line); qa[r["id"]] = r
    md = build(qa, latest("answers_baseline"), latest("retrieval_rerank"), latest("ret_retrievalhybrid"),
               latest("routing_baseline"), latest("ablation_full"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT} ({md.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
