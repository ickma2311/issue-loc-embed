"""Paired comparisons for the sentence-transformers evaluations (st_eval_*.json from eval_st.py). Pure CPU, seconds.

Paired bootstrap as judge.py (10,000 resamples, numpy default_rng(0), paired by instance_id).
  Lite      REL - B0 at Acc@5/10/20 (the card's CIs now come from this run); REL - REF uses the original
            SweRankEmbed-Small per-task file (reference, not re-run); agreement with the original per-task
            results (eval_M2.json / eval_B0_R1.json) and the 1.5-point reproduction tolerance
  LocBench  REL - B0 function Acc@10 on the headline subset (sktime and Qiskit excluded), with its CI;
            also Acc@5/10/15, file Acc@5/10, subsets all / excl. sktime, REL - QWEN06, REL - REF
  Kept list count vs 43,068 (tolerance ±1%), sha256 of the sorted kept ids
Usage: python compare_st.py --dir <eval_st outputs> --orig <earlier eval_*.json dir> [--out comparisons_st.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LITE_FINGERPRINT = "be1407181ff30bc2"  # gpu_run.fingerprint of the evaluated SWE-Bench-Lite files
EXCL = {"all": set(), "excl_sktime": {"sktime/sktime"}, "headline": {"sktime/sktime", "qiskit/qiskit"},
        # sensitivity only: matchms/matchms-backup is a copy of the training repository matchms/matchms
        "headline_excl_matchms_backup": {"sktime/sktime", "qiskit/qiskit", "matchms/matchms-backup"}}


def paired(a: dict, b: dict, n=10000, seed=0):
    ids = sorted(a)
    assert ids == sorted(b), "instance sets differ"
    x = np.array([a[i] - b[i] for i in ids], dtype=np.float64)
    boots = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(1) * 100
    return {"diff": round(float(x.mean() * 100), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2),
                                                             round(float(np.percentile(boots, 97.5)), 2)], "n": len(ids)}


def hits(res, k, key="func_need_k", drop=frozenset()):
    return {p["instance_id"]: int(p[key] <= k) for p in res["per_instance"]
            if p.get("repo", "").lower() not in drop}


def acc(res, k, key="func_need_k", drop=frozenset()):
    h = hits(res, k, key, drop)
    return round(100 * sum(h.values()) / len(h), 2)


def load(d: Path, name: str):
    f = d / name
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    ids = [p["instance_id"] for p in r.get("per_instance", [])]
    if name.startswith("st_eval_"):
        assert not r.get("smoke"), f"{name}: smoke-test result (--limit / --max-units), not a full run"
        n_expected = {"lite": 274, "locb": 560}[r["set"]]
    elif name.startswith("eval_"):  # earlier gpu_run.py evaluation (SWE-Bench-Lite, paper protocol)
        assert r.get("eval_set") == "paper", f"{name}: eval_set {r.get('eval_set')!r}, expected 'paper'"
        assert r.get("data_fingerprint") == LITE_FINGERPRINT, f"{name}: data fingerprint {r.get('data_fingerprint')}"
        n_expected = 274
    else:
        return r
    assert len(ids) == len(set(ids)), f"{name}: duplicate instance ids"
    assert len(ids) == n_expected and r.get("n", len(ids)) == len(ids), f"{name}: {len(ids)} tasks, expected {n_expected}"
    return r


def same_inputs(results: dict, what: str):
    """Every model of one benchmark must have been scored on byte-identical input files."""
    hs = {m: tuple(sorted(r["inputs_sha256"].items())) for m, r in results.items() if r}
    assert len(set(hs.values())) <= 1, f"{what}: input files differ between models {hs}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, o = Path(a.dir), Path(a.orig)
    out = {}
    L = {m: load(d, f"st_eval_lite_{m}.json") for m in ("REL", "B0")}
    same_inputs(L, "lite")
    orig = {"REL": load(o, "eval_M2.json"), "B0": load(o, "eval_B0_R1.json"), "REF": load(o, "eval_REF_R1.json")}
    if all(L.values()):
        lite = {"acc": {m: {k: acc(r, k) for k in (1, 5, 10, 20)} for m, r in L.items()},
                "file_acc1": {m: acc(r, 1, "file_need_k") for m, r in L.items()},
                "REL_minus_B0": {k: paired(hits(L["REL"], k), hits(L["B0"], k)) for k in (5, 10, 20)},
                "REL_minus_REF_original": {k: paired(hits(L["REL"], k), hits(orig["REF"], k)) for k in (5, 10, 20)},
                "reproduction": {}}
        for m in ("REL", "B0"):
            po = {p["instance_id"]: p["func_need_k"] for p in orig[m]["per_instance"]}
            pn = {p["instance_id"]: p["func_need_k"] for p in L[m]["per_instance"]}
            same10 = sum((po[i] <= 10) == (pn[i] <= 10) for i in po)
            lite["reproduction"][m] = {"orig_acc10": orig[m]["func_acc"]["10"], "new_acc10": lite["acc"][m][10],
                                       "within_1.5": abs(orig[m]["func_acc"]["10"] - lite["acc"][m][10]) <= 1.5,
                                       "tasks_same_hit_at10": same10, "tasks_same_rank": sum(po[i] == pn[i] for i in po),
                                       "n": len(po)}
        out["lite"] = lite
    B = {m: load(d, f"st_eval_locb_{m}.json") for m in ("REL", "B0", "QWEN06", "REF")}
    same_inputs(B, "locb")
    if B["REL"] and B["B0"]:
        lb = {"acc": {}, "file_acc": {}, "pairs": {}}
        for sub, drop in EXCL.items():
            lb["acc"][sub] = {m: {k: acc(r, k, drop=drop) for k in (5, 10, 15)} for m, r in B.items() if r}
            lb["file_acc"][sub] = {m: {k: acc(r, k, "file_need_k", drop) for k in (5, 10)} for m, r in B.items() if r}
            lb["pairs"][sub] = {f"REL_minus_{m}": {k: paired(hits(B["REL"], k, drop=drop), hits(B[m], k, drop=drop))
                                                   for k in (5, 10, 15)} for m in ("B0", "QWEN06", "REF") if B[m]}
        g = lb["pairs"]["headline"]["REL_minus_B0"][10]
        lb["headline"] = {"comparison": "REL - B0, function Acc@10, headline subset, paired 95% CI",
                      **g, "ci_excludes_zero": g["ci95"][0] > 0}
        lb["protocol_selfcheck"] = {"B0_acc10_all": lb["acc"]["all"]["B0"][10], "paper_B0": 47.86,
                                    "flag": abs(lb["acc"]["all"]["B0"][10] - 47.86) > 5}
        if B["REF"]:
            lb["protocol_selfcheck"].update(REF_acc10_all=lb["acc"]["all"]["REF"][10], paper_REF=58.57,
                                            flag_REF=abs(lb["acc"]["all"]["REF"][10] - 58.57) > 5)
        out["locbench"] = lb
    kept = load(d, "kept_it1.json")
    if kept:
        ids = sorted(kept["kept"])
        out["kept_list"] = {"count": len(ids), "original": 43068, "within_1pct": abs(len(ids) - 43068) <= 430,
                            "sha256_sorted_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                            "mine_stats": load(d, "mine_it1.json"), "original_mine_stats": load(o, "mine_it1.json")}
    s = json.dumps(out, indent=1)
    Path(a.out or d / "comparisons_st.json").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
