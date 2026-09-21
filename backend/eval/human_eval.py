"""
Human evaluation: a blind rating sheet from real system outputs, and a scorer.

    cd backend
    python -m eval.human_eval make-sheet --n 20 --configs dense,hybrid+rerank,full_e2e
        -> eval/human/HUMAN_EVAL_SHEET.csv   (give one copy to each rater)
        -> eval/human/key.json               (which configuration each row came from; keep hidden)
    python -m eval.human_eval score --sheets rater1.csv rater2.csv
        -> per-configuration mean ratings, inter-rater agreement, and agreement with contains-gold

Raters see question + answer only. They fill two columns, 1-5 each:
  correctness  (1 wrong … 5 fully correct)   groundedness (1 invented … 5 clearly from the sources)
Answers come from the ablation run's per-item outputs, so they are exactly what the
system produced, shuffled so configurations cannot be told apart by position.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
HUMAN = os.path.join(HERE, "human")
COLS = ["row", "id", "question", "answer", "correctness_1_5", "groundedness_1_5", "comment"]


def _latest(prefix):
    fs = sorted(glob.glob(os.path.join(RESULTS, f"{prefix}_*.json")), key=os.path.getmtime)
    return json.load(open(fs[-1], encoding="utf-8")) if fs else None


def make_sheet(n: int, configs: list[str], seed: int = 7) -> tuple[list[dict], dict]:
    qa = {}
    with open(os.path.join(HERE, "datasets", "qa.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                r = json.loads(line); qa[r["id"]] = r
    per_item = {}
    for src in ("ablation_full", "ablation_e2e_probe", "ablation_e2e"):
        d = _latest(src)
        if d:
            for cfg, rows in d["per_item"].items():
                per_item.setdefault(cfg, {r["id"]: r for r in rows})
    missing = [c for c in configs if c not in per_item]
    if missing:
        raise SystemExit(f"no results for configurations {missing}; run eval.ablation first")
    rng = random.Random(seed)
    ids = sorted(set.intersection(*[set(per_item[c]) for c in configs]))
    rng.shuffle(ids); ids = ids[:n]
    rows = []
    for i in ids:
        for c in configs:
            r = per_item[c][i]
            rows.append({"id": i, "question": qa[i]["question"], "answer": r["answer"], "config": c,
                         "contains_gold": r["contains_gold"]})
    rng.shuffle(rows)
    sheet, key = [], {}
    for k, r in enumerate(rows, start=1):
        sheet.append({"row": k, "id": r["id"], "question": r["question"], "answer": r["answer"],
                      "correctness_1_5": "", "groundedness_1_5": "", "comment": ""})
        key[str(k)] = {"config": r["config"], "contains_gold": r["contains_gold"]}
    return sheet, key


def _read_sheet(path: str) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[str(r["row"])] = r
    return out


def _num(v):
    try:
        x = float(v); return x if 1 <= x <= 5 else None
    except (TypeError, ValueError):
        return None


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def score(sheet_paths: list[str], key: dict) -> dict:
    sheets = [_read_sheet(p) for p in sheet_paths]
    per_cfg = defaultdict(lambda: {"correctness": [], "groundedness": [], "contains_gold": []})
    agree_pairs = {"correctness": ([], []), "groundedness": ([], [])}
    human_vs_auto = ([], [])
    for row, meta in key.items():
        vals = {"correctness": [], "groundedness": []}
        for s in sheets:
            r = s.get(row)
            if not r:
                continue
            for m, col in (("correctness", "correctness_1_5"), ("groundedness", "groundedness_1_5")):
                v = _num(r.get(col))
                if v is not None:
                    vals[m].append(v)
        for m in vals:
            if vals[m]:
                per_cfg[meta["config"]][m].append(sum(vals[m]) / len(vals[m]))
            if len(vals[m]) >= 2:
                agree_pairs[m][0].append(vals[m][0]); agree_pairs[m][1].append(vals[m][1])
        if vals["correctness"]:
            per_cfg[meta["config"]]["contains_gold"].append(meta["contains_gold"])
            human_vs_auto[0].append(sum(vals["correctness"]) / len(vals["correctness"])); human_vs_auto[1].append(meta["contains_gold"])
    out = {"raters": len(sheets), "per_configuration": {}, "inter_rater": {}, "human_correctness_vs_contains_gold_spearman": _spearman(*human_vs_auto)}
    for cfg, d in per_cfg.items():
        out["per_configuration"][cfg] = {"n": len(d["correctness"]),
                                         "correctness_mean": round(sum(d["correctness"]) / len(d["correctness"]), 2) if d["correctness"] else None,
                                         "groundedness_mean": round(sum(d["groundedness"]) / len(d["groundedness"]), 2) if d["groundedness"] else None,
                                         "contains_gold_rate": round(sum(d["contains_gold"]) / len(d["contains_gold"]), 3) if d["contains_gold"] else None}
    for m, (a, b) in agree_pairs.items():
        if a:
            exact = sum(x == y for x, y in zip(a, b)) / len(a)
            within1 = sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
            out["inter_rater"][m] = {"pairs": len(a), "exact_agreement": round(exact, 3), "within_one_point": round(within1, 3), "spearman": _spearman(a, b)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sheet"); m.add_argument("--n", type=int, default=20); m.add_argument("--configs", default="dense,hybrid+rerank,full_e2e"); m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score"); s.add_argument("--sheets", nargs="+", required=True); s.add_argument("--key", default=os.path.join(HUMAN, "key.json"))
    args = ap.parse_args()
    os.makedirs(HUMAN, exist_ok=True)
    if args.cmd == "make-sheet":
        sheet, key = make_sheet(args.n, args.configs.split(","), args.seed)
        path = os.path.join(HUMAN, "HUMAN_EVAL_SHEET.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(sheet)
        with open(os.path.join(HUMAN, "key.json"), "w", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
        print(f"wrote {path} ({len(sheet)} rows = {args.n} questions × {len(args.configs.split(','))} configurations) and key.json (keep hidden from raters)")
    else:
        key = json.load(open(args.key, encoding="utf-8"))
        print(json.dumps(score(args.sheets, key), indent=2))


if __name__ == "__main__":
    main()
