"""Paired comparisons on the SWE-Bench-Lite eval_*.json files written by gpu_run.py. Pure CPU, seconds.

Paired bootstrap over issues (10,000 resamples, numpy default_rng(0), paired by instance_id) for
M2 - B0, M1 - A1 and M1 - B0 at function Acc@10. B0 is taken in the representation chosen by calibration
(rep.json: the one where B0 is stronger).
Usage: python judge.py --dir <folder with eval_*.json and rep.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, name: str):
    f = d / f"eval_{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def hits(res, k=10):
    return {p["instance_id"]: int(p["func_need_k"] <= k) for p in res["per_instance"]}


def paired(a, b, n=10000, seed=0):
    ids = sorted(a)
    assert ids == sorted(b), "instance sets differ"
    x = np.array([a[i] - b[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1) * 100
    return {"diff": float(x.mean() * 100), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rep = json.loads((d / "rep.json").read_text())["rep"]
    R = {k: load(d, k) for k in ("M2", "M1", "A1", "B0_" + rep)}
    R["B0"] = R.pop("B0_" + rep)
    fps = {k: v.get("data_fingerprint") for k, v in R.items() if v}
    assert len(set(fps.values())) == 1, f"eval data differ between runs: {fps}"
    out = {"rep": rep, "acc10": {k: v["func_acc"]["10"] if "10" in v["func_acc"] else v["func_acc"][10]
                                 for k, v in R.items() if v}, "fingerprints": fps}
    for x, y in (("M2", "B0"), ("M1", "A1"), ("M1", "B0")):
        if R[x] and R[y]:
            out[f"{x}_minus_{y}"] = paired(hits(R[x]), hits(R[y]))
    (d / "comparisons.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
