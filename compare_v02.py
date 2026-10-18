"""Compare v0.2 with v0.1 and CodeRankEmbed on SWE-bench Multilingual, SWE-Bench-Lite and LocBench (the numbers in the
model card). Inputs: one directory with eval_st.py outputs st_eval_{ml,lite,locb}_{V02,REL,B0}.json, and the
Multilingual task file (build_multilang.py output, for the language of each issue).
Acc@k = all gold functions in the top k. Paired bootstrap over issues: 10,000 resamples, numpy default_rng(0).
LocBench "main" leaves out sktime/sktime and Qiskit/qiskit (training contains a fork and a renamed copy).
Usage: python -I compare_v02.py <dir> <ml_tasks.parquet> > v02_results.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KS = (1, 5, 10)
LOCB_EXCL = {"sktime/sktime", "qiskit/qiskit"}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    b = x[np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))].mean(1) * 100
    return {"diff": round(float(x.mean() * 100), 2),
            "ci95": [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]}


def table(models, ids):
    row = {"n": len(ids)}
    for k in KS:
        hit = {m: [int(models[m][i]["func_need_k"] <= k) for i in ids] for m in models}
        row[f"acc{k}"] = {m: round(100 * float(np.mean(h)), 2) for m, h in hit.items()}
        row[f"acc{k}"]["v02_minus_v01"] = boot([a - b for a, b in zip(hit["V02"], hit["REL"])])
        row[f"acc{k}"]["v02_minus_coderankembed"] = boot([a - b for a, b in zip(hit["V02"], hit["B0"])])
    return row


def main(d, ml_tasks):
    d = Path(d)
    lang = {t["instance_id"]: t for t in pq.read_table(ml_tasks, columns=["instance_id", "language", "language_group"]).to_pylist()}
    out = {}
    for s in ("ml", "lite", "locb"):
        models = {m: {p["instance_id"]: p for p in json.load(open(d / f"st_eval_{s}_{m}.json"))["per_instance"]}
                  for m in ("V02", "REL", "B0")}
        ids = sorted(models["V02"])
        assert set(ids) == set(models["REL"]) == set(models["B0"]), s
        res = {"all": table(models, ids)}
        if s == "ml":
            for key in ("language_group", "language"):
                for L in sorted({lang[i][key] for i in ids}):
                    res[f"{key}:{L}"] = table(models, [i for i in ids if lang[i][key] == L])
        if s == "locb":
            res["main"] = table(models, [i for i in ids if models["V02"][i]["repo"].lower() not in LOCB_EXCL])
        out[s] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
