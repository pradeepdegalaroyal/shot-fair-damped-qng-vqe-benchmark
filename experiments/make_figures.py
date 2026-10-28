"""Rebuild the summary tables and figures from results/runs.csv without rerunning any experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shotfair_qng.plotting import make_all_figures  # noqa: E402

if __name__ == "__main__":
    res = ROOT / "results"
    df = pd.read_csv(res / "runs.csv")
    refs = {int(k): v for k, v in json.load(open(res / "reference.json"))["reference"].items()}
    make_all_figures(df, refs, ROOT / "figures", res)
    print("figures and tables rebuilt")
