"""Significance statistics, pinned to values that can be checked by hand or against a textbook."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.significance import _latest, build, compare, identical_share, mcnemar_exact, paired_bootstrap_ci, questions_needed, wilson


def test_wilson_matches_known_values_and_behaves_at_the_edges():
    lo, hi = wilson(47, 64)                       # 0.734
    assert round(lo, 3) == 0.615 and round(hi, 3) == 0.827
    lo, hi = wilson(0, 64)
    assert lo == 0.0 and 0.0 < hi < 0.06, "zero successes still has an upper bound"
    lo, hi = wilson(64, 64)
    assert hi == 1.0 and 0.94 < lo < 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert math.isclose(mcnemar_exact(10, 0), 2 / 2 ** 10)                 # 0.00195
    assert math.isclose(mcnemar_exact(8, 2), 2 * (1 + 10 + 45) / 2 ** 10)  # 0.109
    assert mcnemar_exact(8, 2) == mcnemar_exact(2, 8), "two-sided: direction does not matter"
    assert mcnemar_exact(6, 1) > 0.05 > mcnemar_exact(9, 1)


def test_paired_bootstrap_is_seeded_and_brackets_the_observed_difference():
    a = [0.0] * 30 + [1.0] * 34
    b = [1.0] * 20 + [0.0] * 10 + [1.0] * 34          # 20 gained, 0 lost -> delta +0.3125
    lo, hi = paired_bootstrap_ci(a, b)
    assert (lo, hi) == paired_bootstrap_ci(a, b), "same seed, same interval"
    assert 0.18 < lo < 0.3125 < hi < 0.45
    lo0, hi0 = paired_bootstrap_ci(a, a)
    assert lo0 == hi0 == 0.0


def test_questions_needed_grows_as_the_effect_shrinks():
    assert questions_needed(5, 5, 64) is None, "no net effect: no sample size confirms it"
    big, small = questions_needed(20, 2, 64), questions_needed(8, 4, 64)
    assert big < 64 < small, "a large effect is confirmable within the current set; a small one is not"
    assert questions_needed(10, 0, 64) >= 1


def test_compare_counts_only_shared_questions_and_reports_both_directions():
    a = {"q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 1.0, "only_a": 1.0}
    b = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.0, "only_b": 0.0}
    c = compare(a, b)
    assert c["n"] == 4 and c["gained"] == 2 and c["lost"] == 1
    assert c["acc_a"] == 0.5 and c["acc_b"] == 0.75 and c["delta"] == 0.25
    assert c["flip_rate"] == 0.75 and c["significant"] is False


def test_report_separates_established_from_not_established_and_includes_the_noise_floor():
    ids = [f"q{i}" for i in range(64)]
    none = {i: 0.0 for i in ids}
    dense = {i: float(k < 41) for k, i in enumerate(ids)}
    stack = {i: float(3 <= k < 50) for k, i in enumerate(ids)}   # vs dense: 9 gained, 3 lost -> 47/64, p 0.146
    rep = {i: float(3 <= k < 48 or k == 60) for k, i in enumerate(ids)}   # vs stack: 1 gained, 2 lost -> 3 flips
    md = build({"no_retrieval": none, "dense": dense, "hybrid+rerank": stack, "hybrid+rerank, repeat run 2": rep})
    assert "| no_retrieval | 0/64 | 0.000 |" in md and "| hybrid+rerank | 47/64 | 0.734 | 0.615 – 0.827 |" in md
    assert "is retrieval worth anything?" in md and "**significant**" in md
    assert "Supported by this evaluation" in md and "Not established at n = 64" in md
    assert "whole shipped retrieval stack" in md and "questions would settle it" in md
    assert "## 3. The noise floor" in md and "3/64" in md
    assert "repeat run" not in md.split("## 2.")[0], "repeat runs are not listed as configurations"


def test_latest_matches_the_exact_label_not_a_longer_one(tmp_path):
    """Regression: "ablation_e2e_*" also matched "ablation_e2e_probe_*", so the before/after routing
    comparison silently compared a run with itself (0 gained, 0 lost) instead of 0.094 vs 0.672."""
    import json, os, time
    before = tmp_path / "ablation_e2e_20260921_104854.json"; before.write_text(json.dumps({"which": "before"}))
    time.sleep(0.01)
    after = tmp_path / "ablation_e2e_probe_20260921_110226.json"; after.write_text(json.dumps({"which": "after"}))
    os.utime(after, None)                                   # the probe file is the newer one
    assert _latest("ablation_e2e", str(tmp_path)) == {"which": "before"}
    assert _latest("ablation_e2e_probe", str(tmp_path)) == {"which": "after"}
    assert _latest("ablation", str(tmp_path)) is None and _latest("missing", str(tmp_path)) is None


def test_identical_share_ignores_truncation_and_whitespace_but_not_content():
    a = {"q1": "FAISS is the vector database. " + "x" * 600, "q2": "Ollama hosts it.", "q3": "same"}
    b = {"q1": "FAISS is the vector database. " + "x" * 450, "q2": "Ollama serves it.", "q3": "same \n"}
    assert identical_share(a, b) == 2 / 3, "q1 differs only beyond the compared window; q2 differs in content"
    assert identical_share(a, {}) is None


def test_report_recommends_greedy_only_when_it_actually_reduces_flips():
    ids = [f"q{i}" for i in range(64)]
    base = {i: float(k < 47) for k, i in enumerate(ids)}
    noisy = {i: float(k < 43 or k >= 60) for k, i in enumerate(ids)}          # 8 flips vs base
    series = {"hybrid+rerank": base, "hybrid+rerank, repeat run 2": noisy,
              "hybrid+rerank, greedy run 1": base, "hybrid+rerank, greedy run 2": dict(base)}
    answers = {k: {i: f"answer {i}" for i in ids} for k in series}
    md = build(series, answers)
    assert "### Greedy decoding" in md and "0 of 64 verdicts flipped" in md and "100.0% of answers were character-for-character identical" in md
    assert "Evaluation runs should set `OLLAMA_TEMPERATURE=0`" in md
    series["hybrid+rerank, greedy run 2"] = noisy                                # greedy no better than sampling
    assert "did not remove the noise" in build(series, answers)
