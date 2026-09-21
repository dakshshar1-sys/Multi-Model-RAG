"""Human-evaluation tooling: blind sheet from real outputs; scorer with agreement."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.human_eval import COLS, _spearman, score


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def test_score_means_agreement_and_correlation(tmp_path):
    key = {"1": {"config": "dense", "contains_gold": 0.0}, "2": {"config": "hybrid+rerank", "contains_gold": 1.0},
           "3": {"config": "dense", "contains_gold": 1.0}, "4": {"config": "hybrid+rerank", "contains_gold": 1.0}}
    base = [{"row": i, "id": f"q{i}", "question": "q", "answer": "a", "comment": ""} for i in range(1, 5)]
    r1 = [dict(b, correctness_1_5=c, groundedness_1_5=g) for b, c, g in zip(base, [2, 5, 4, 5], [1, 5, 4, 4])]
    r2 = [dict(b, correctness_1_5=c, groundedness_1_5=g) for b, c, g in zip(base, [1, 5, 4, 4], [2, 5, 5, 4])]
    p1, p2 = tmp_path / "r1.csv", tmp_path / "r2.csv"; _write(p1, r1); _write(p2, r2)
    out = score([str(p1), str(p2)], key)
    assert out["raters"] == 2
    assert out["per_configuration"]["hybrid+rerank"]["correctness_mean"] > out["per_configuration"]["dense"]["correctness_mean"]
    assert out["per_configuration"]["dense"]["n"] == 2 and out["per_configuration"]["dense"]["contains_gold_rate"] == 0.5
    ir = out["inter_rater"]["correctness"]
    assert ir["pairs"] == 4 and 0 <= ir["exact_agreement"] <= 1 and ir["within_one_point"] == 1.0
    assert out["human_correctness_vs_contains_gold_spearman"] > 0.5


def test_blank_and_out_of_range_ratings_are_ignored(tmp_path):
    key = {"1": {"config": "dense", "contains_gold": 1.0}, "2": {"config": "dense", "contains_gold": 0.0}}
    rows = [{"row": 1, "id": "q1", "question": "q", "answer": "a", "correctness_1_5": "", "groundedness_1_5": "9", "comment": ""},
            {"row": 2, "id": "q2", "question": "q", "answer": "a", "correctness_1_5": "3", "groundedness_1_5": "x", "comment": ""}]
    p = tmp_path / "r.csv"; _write(p, rows)
    out = score([str(p)], key)
    assert out["per_configuration"]["dense"]["n"] == 1 and out["per_configuration"]["dense"]["groundedness_mean"] is None


def test_spearman_handles_ties_and_short_input():
    assert _spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert _spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert _spearman([1, 1, 2], [1, 1, 2]) == 1.0
    assert _spearman([1, 2], [1, 2]) is None
