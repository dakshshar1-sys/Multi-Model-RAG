"""One place that knows how result files are named: <label>_<YYYYMMDD>_<HHMMSS>.json in eval/results/."""
from __future__ import annotations

import glob
import json
import os
import re

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_STAMP = re.compile(r"_\d{8}_\d{6}\.json$")


def latest_result(label: str, results_dir: str = RESULTS) -> dict | None:
    """Newest results file whose label is EXACTLY `label`. A plain glob on "ablation_e2e_*" also
    matches "ablation_e2e_probe_*" and silently hands back the wrong run."""
    files = [f for f in glob.glob(os.path.join(results_dir, f"{label}_*.json"))
             if _STAMP.sub("", os.path.basename(f)) == label]
    files.sort(key=os.path.getmtime)
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)
