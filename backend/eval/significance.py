"""
Are the evaluation's differences real? Confidence intervals and paired tests, from results on disk.

    cd backend && python -m eval.significance          # -> eval/SIGNIFICANCE.md   (no model calls)

BASELINE.md reports point estimates on 64 questions ("hybrid + rerank 0.734 vs dense 0.641").
With n = 64 one question is 1.6 points, and generation is sampled, so the first thing an examiner
asks is whether a gap is signal or noise. Every configuration answered the SAME questions, so the
right tools are paired:

  Wilson interval       95% CI for each configuration's accuracy (well-behaved near 0 and 1)
  exact McNemar test    looks only at questions where two configurations DISAGREE: `gained`
                        (A wrong, B right) vs `lost` (A right, B wrong); two-sided binomial p
  paired bootstrap      95% CI for the accuracy difference, resampling questions (seeded)
  questions needed      Connor's approximation: how many questions would confirm a gain of this
                        size at alpha 0.05 with 80% power - i.e. how big the next question set must be
  noise floor           the identical configuration run three times: accuracy spread and the share
                        of questions whose verdict flipped, with no change to the system at all

Pure Python on purpose (no SciPy): the numbers must be reproducible inside the CI environment.
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.results_io import RESULTS, latest_result  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SIGNIFICANCE.md")
Z95, Z80 = 1.959964, 0.841621
ALPHA = 0.05


# ── statistics ───────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value: under H0 each discordant question is a fair coin."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(a: list[float], b: list[float], reps: int = 10000, seed: int = 7) -> tuple[float, float]:
    """95% percentile CI for mean(b) - mean(a), resampling questions (pairs stay together)."""
    assert len(a) == len(b) and a
    rng, n = random.Random(seed), len(a)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return (means[int(0.025 * reps)], means[int(0.975 * reps) - 1])


def questions_needed(gained: int, lost: int, n: int, z_a: float = Z95, z_b: float = Z80) -> int | None:
    """Connor (1987): pairs needed for McNemar to detect this effect, alpha 0.05 two-sided, power 0.8."""
    if n == 0 or gained == lost:
        return None
    psi, d = (gained + lost) / n, (gained - lost) / n
    inner = psi - d * d
    if inner <= 0:
        return max(1, math.ceil((z_a * math.sqrt(psi)) ** 2 / (d * d)))
    return math.ceil((z_a * math.sqrt(psi) + z_b * math.sqrt(inner)) ** 2 / (d * d))


def compare(a: dict[str, float], b: dict[str, float]) -> dict:
    """Paired comparison of two configurations given {question id: 0/1}. Only shared ids count."""
    ids = sorted(set(a) & set(b))
    xa, xb = [float(a[i]) for i in ids], [float(b[i]) for i in ids]
    gained = sum(1 for x, y in zip(xa, xb) if not x and y)
    lost = sum(1 for x, y in zip(xa, xb) if x and not y)
    n = len(ids)
    lo, hi = paired_bootstrap_ci(xa, xb) if n else (0.0, 0.0)
    p = mcnemar_exact(gained, lost)
    return {"n": n, "acc_a": sum(xa) / n if n else 0.0, "acc_b": sum(xb) / n if n else 0.0,
            "gained": gained, "lost": lost, "delta": (sum(xb) - sum(xa)) / n if n else 0.0,
            "ci": (lo, hi), "p": p, "significant": p < ALPHA,
            "questions_needed": questions_needed(gained, lost, n),
            "flip_rate": (gained + lost) / n if n else 0.0}


# ── loading ──────────────────────────────────────────────────────────────────
def _latest(prefix: str, results_dir: str = RESULTS) -> dict | None:
    return latest_result(prefix, results_dir)


def _items(rows: list[dict], key: str = "contains_gold") -> dict[str, float]:
    return {r["id"]: float(bool(r.get(key))) for r in rows}


def load_series(results_dir: str = RESULTS) -> dict[str, dict[str, float]]:
    """Every per-question 0/1 series the harness has produced, keyed by a readable name."""
    s: dict[str, dict[str, float]] = {}
    full = _latest("ablation_full", results_dir)
    if full:
        for name, rows in full["per_item"].items():
            s[name] = _items(rows)
    for prefix, name in (("ablation_e2e", "end-to-end, before retrieval-first routing"),
                         ("ablation_e2e_probe", "end-to-end, with retrieval-first routing")):
        d = _latest(prefix, results_dir)
        if d:
            s[name] = _items(d["per_item"]["full_e2e"])
    ll = _latest("ablation_llama32", results_dir)
    if ll and "hybrid+rerank" in ll["per_item"]:
        s["hybrid+rerank (llama3.2)"] = _items(ll["per_item"]["hybrid+rerank"])
    for prefix, name in (("abstention_before", "hybrid+rerank, repeat run 2"), ("abstention_after", "hybrid+rerank, repeat run 3")):
        d = _latest(prefix, results_dir)
        if d and d.get("answerable"):
            s[name] = _items(d["answerable"])
    for k in (1, 2):
        d = _latest(f"ablation_greedy{k}", results_dir)
        if d and "hybrid+rerank" in d["per_item"]:
            s[f"hybrid+rerank, greedy run {k}"] = _items(d["per_item"]["hybrid+rerank"])
    for prefix, name in (("retrieval_rerank", "retrieval recall@5: dense"), ("ret_retrievalhybrid", "retrieval recall@5: hybrid")):
        d = _latest(prefix, results_dir)
        if d:
            s[name] = _items(d["retrieval"]["rows"], key="pre_recall@5")
    return s


def load_answers(results_dir: str = RESULTS) -> dict[str, dict[str, str]]:
    """Answer text per question for the repeated runs, to count character-identical answers."""
    a: dict[str, dict[str, str]] = {}
    full = _latest("ablation_full", results_dir)
    if full and "hybrid+rerank" in full["per_item"]:
        a["hybrid+rerank"] = {r["id"]: r.get("answer", "") for r in full["per_item"]["hybrid+rerank"]}
    for prefix, name in (("abstention_before", "hybrid+rerank, repeat run 2"), ("abstention_after", "hybrid+rerank, repeat run 3")):
        d = _latest(prefix, results_dir)
        if d and d.get("answerable"):
            a[name] = {r["id"]: r.get("answer", "") for r in d["answerable"]}
    for k in (1, 2):
        d = _latest(f"ablation_greedy{k}", results_dir)
        if d and "hybrid+rerank" in d["per_item"]:
            a[f"hybrid+rerank, greedy run {k}"] = {r["id"]: r.get("answer", "") for r in d["per_item"]["hybrid+rerank"]}
    return a


def identical_share(a: dict[str, str], b: dict[str, str], chars: int = 400) -> float | None:
    """Share of shared questions whose answers match character for character (first `chars`;
    result files store a truncated answer, and runs truncate at different lengths)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return sum(1 for i in ids if a[i][:chars].strip() == b[i][:chars].strip()) / len(ids)


COMPARISONS = [  # (A, B, what the comparison asks)
    ("no_retrieval", "dense", "is retrieval worth anything?"),
    ("dense", "dense+rerank", "the reranker, on dense"),
    ("dense", "hybrid", "lexical retrieval added to dense"),
    ("hybrid", "hybrid+rerank", "the reranker, on hybrid"),
    ("dense", "hybrid+rerank", "the whole shipped retrieval stack vs baseline RAG"),
    ("end-to-end, before retrieval-first routing", "end-to-end, with retrieval-first routing", "the routing fix, end to end"),
    ("hybrid+rerank", "hybrid+rerank (llama3.2)", "qwen2.5:3b vs llama3.2, same stack"),
    ("retrieval recall@5: dense", "retrieval recall@5: hybrid", "hybrid retrieval, measured at retrieval (no generation noise)"),
    ("hybrid+rerank", "hybrid+rerank, greedy run 1", "does greedy decoding (temperature 0) cost accuracy?"),
]
NOISE = [("hybrid+rerank", "hybrid+rerank, repeat run 2"), ("hybrid+rerank", "hybrid+rerank, repeat run 3"),
         ("hybrid+rerank, repeat run 2", "hybrid+rerank, repeat run 3")]
NOISE_GREEDY = [("hybrid+rerank, greedy run 1", "hybrid+rerank, greedy run 2")]


# ── report ───────────────────────────────────────────────────────────────────
def _pct(x: float) -> str:
    return f"{x:.3f}"


def _p(p: float) -> str:
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def _noise_rows(pairs, series, answers) -> tuple[list[str], list[dict]]:
    rows, stats = [], []
    for a, b in pairs:
        if a not in series or b not in series:
            continue
        c = compare(series[a], series[b])
        same = identical_share(answers.get(a, {}), answers.get(b, {})) if answers else None
        c["identical"] = same
        rows.append(f"| {a} → {b} | {_pct(c['acc_a'])} | {_pct(c['acc_b'])} | {c['gained'] + c['lost']}/{c['n']} ({c['flip_rate']:.1%}) | "
                    f"{'–' if same is None else f'{same:.1%}'} | {_p(c['p'])} |")
        stats.append(c)
    return rows, stats


def build(series: dict[str, dict[str, float]], answers: dict[str, dict[str, str]] | None = None) -> str:
    out = ["# Statistical significance of the evaluation (generated)\n",
           "Produced by `python -m eval.significance` from the per-question results in `eval/results/`. No model is",
           "called; nothing here is edited by hand. Method and formulas: the docstring of `eval/significance.py`.\n",
           "## 1. Each configuration, with a 95% confidence interval\n",
           "Wilson intervals. With 64 questions the interval is about ±0.11 wide, so two configurations can differ by",
           "several points and still overlap: **overlap is not a test**, the paired comparison in section 2 is.\n",
           "| configuration | correct | accuracy | 95% CI |", "|---|---|---|---|"]
    for name, items in series.items():
        if "repeat run" in name or "greedy run 2" in name:
            continue
        k, n = int(sum(items.values())), len(items)
        lo, hi = wilson(k, n)
        out.append(f"| {name} | {k}/{n} | {_pct(k / n)} | {_pct(lo)} – {_pct(hi)} |")

    out += ["\n## 2. Paired comparisons (same questions, exact McNemar)\n",
            "`gained` = questions B got right that A got wrong; `lost` = the reverse. Only those questions carry",
            "information. `questions needed` = size of a question set that would confirm a gain of this size",
            "(alpha 0.05, power 0.8).\n",
            "| question | A → B | A | B | gained | lost | Δ | 95% CI of Δ | p | verdict | questions needed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    sig, not_sig = [], []
    for a, b, ask in COMPARISONS:
        if a not in series or b not in series:
            continue
        c = compare(series[a], series[b])
        verdict = "**significant**" if c["significant"] else "not significant at this n"
        need = "–" if c["significant"] or c["questions_needed"] is None else f"≈ {c['questions_needed']}"
        out.append(f"| {ask} | {a} → {b} | {_pct(c['acc_a'])} | {_pct(c['acc_b'])} | {c['gained']} | {c['lost']} | {c['delta']:+.3f} | "
                   f"{c['ci'][0]:+.3f} – {c['ci'][1]:+.3f} | {_p(c['p'])} | {verdict} | {need} |")
        (sig if c["significant"] else not_sig).append((ask, c))

    answers = answers or {}
    rows, stats = _noise_rows(NOISE, series, answers)
    if rows:
        out += ["\n## 3. The noise floor: the identical configuration, run repeatedly\n",
                "Same retrieval, same prompt, same model. Any difference here is noise, and it is the yardstick for",
                "every generation-level difference above.\n",
                "### Sampled decoding (the model's default temperature, what production uses)\n",
                "| run A → run B | A | B | verdicts that flipped | identical answers | p |", "|---|---|---|---|---|---|"] + rows
        flips = [c["flip_rate"] for c in stats]
        accs = sorted({round(c["acc_a"], 3) for c in stats} | {round(c["acc_b"], 3) for c in stats})
        out.append(f"\nAccuracy ranged {accs[0]:.3f}–{accs[-1]:.3f} with nothing changed, and on average "
                   f"**{sum(flips) / len(flips):.1%} of questions flipped verdict between two identical runs**. "
                   "None of these pairs is significant, as it should be: the test does not mistake noise for an effect. "
                   "But compare the flip counts with the `gained + lost` counts in section 2: most single-run "
                   "answer-level differences there are no larger than this.")
    grows, gstats = _noise_rows(NOISE_GREEDY, series, answers)
    if grows:
        out += ["\n### Greedy decoding (`OLLAMA_TEMPERATURE=0`, for evaluation runs)\n",
                "| run A → run B | A | B | verdicts that flipped | identical answers | p |", "|---|---|---|---|---|---|"] + grows
        g = gstats[0]
        same = "" if g["identical"] is None else f" and {g['identical']:.1%} of answers were character-for-character identical"
        if stats:
            base = sum(c["flip_rate"] for c in stats) / len(stats)
            out.append(f"\nWith greedy decoding {g['gained'] + g['lost']} of {g['n']} verdicts flipped ({g['flip_rate']:.1%}, against {base:.1%} sampled){same}. "
                       + ("Evaluation runs should set `OLLAMA_TEMPERATURE=0`: a difference between two greedy runs is then a difference between two systems. "
                          if g["flip_rate"] < base / 2 else
                          "Greedy decoding did not remove the noise on this hardware, so repeated runs remain necessary. ")
                       + "Production keeps the model's default sampling; this is an evaluation setting.")

    out.append("\n## 4. What can and cannot be claimed\n")
    if sig:
        out.append("**Supported by this evaluation (p < 0.05):**\n")
        out += [f"- {ask}: {c['acc_a']:.3f} → {c['acc_b']:.3f} ({c['gained']} gained, {c['lost']} lost, p {_p(c['p'])})." for ask, c in sig]
    if not_sig:
        out.append("\n**Not established at n = 64, in either direction:**\n")
        for ask, c in not_sig:
            need = f"; a set of about {c['questions_needed']} questions would settle it" if c["questions_needed"] else ""
            out.append(f"- {ask}: {c['delta']:+.3f} ({c['gained']} gained, {c['lost']} lost, p {_p(c['p'])}){need}.")
        out.append("\nThe honest wording for these is \"consistent with a difference\", not \"differs\". The remedy is more")
        out.append("questions, which is what `datasets/INDEPENDENT_PROTOCOL.md` is for; the `questions needed` column says how many.")
    out.append("\nRetrieval-level metrics (recall@k) involve no sampled generation, so they are the cleaner place to")
    out.append("demonstrate a retrieval change; answer-level metrics inherit the noise in section 3.")
    return "\n".join(out) + "\n"


def main():
    series = load_series()
    if not series:
        print("no result files found in eval/results/: run eval.ablation first"); return
    md = build(series, load_answers())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT} ({len(series)} series)")


if __name__ == "__main__":
    main()
