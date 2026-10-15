"""LocBench eval corpus (Kaggle CPU) with exactly the SWE-Bench-Lite pipeline.

Same code path as the Lite eval set: build_data.process_repo(split="eval") reads every .py file of each
base commit, then issueloc.paperfmt converts to the paper protocol (non-test functions + methods,
"path/Class/func" header, class gold -> Class.__init__).

Outputs ($OUT_DIR/data): locb_tasks.parquet, locb_ent_000.parquet, locb_txt_000.parquet (raw, same layout as the build_data.py eval files),
locb_tasks_paper.parquet, locb_ent_paper_000.parquet, locb_txt_paper_000.parquet, report_locb.json
Usage: python build_locbench.py [--limit-repos N] [--workers W]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

import build_data
from build_data import ShardWriter, log, process_repo
from issueloc.paperfmt import _convert
from issueloc.tasks import s_

LOCB = "czlll/Loc-Bench_V1"
from issueloc.revisions import LOCBENCH_MAIN_REV as LOCB_REV
LOCB_URL = f"https://huggingface.co/datasets/{LOCB}/resolve/{LOCB_REV}/data/test-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-repos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(os.environ.get("OUT_DIR", "out")) / "data"
    out.mkdir(parents=True, exist_ok=True)
    build_data.WORK.mkdir(parents=True, exist_ok=True)
    with fsspec.open(LOCB_URL, "rb") as f:
        rows = pq.read_table(f, columns=["instance_id", "repo", "base_commit", "problem_statement", "patch",
                                         "edit_functions", "category"]).to_pylist()
    assert len(rows) == 560 and all(len(r["edit_functions"]) for r in rows), len(rows)
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append({"instance_id": r["instance_id"], "base_commit": r["base_commit"],
                                   "patch": s_(r["patch"]), "query": s_(r["problem_statement"]),
                                   "gold": list(r["edit_functions"])})
    repos = sorted(by_repo)[: a.limit_repos or None]
    jobs = [{"repo": k, "split": "eval", "tasks": by_repo[k]} for k in repos]
    log(f"LocBench: {sum(len(j['tasks']) for j in jobs)} tasks in {len(jobs)} repos")
    w = ShardWriter(out, "locb", 10**12)
    tasks, fails, errs, done = [], Counter(), [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_repo, j): j for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            fails.update(res["fail"])
            if res.get("err"):
                errs.append((res["repo"], res["err"]))
            tasks += res["tasks"]
            w.add(res)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[progress] {done}/{len(jobs)} repos tasks={len(tasks)} units={len(w.ents)} "
                      f"unique={len(w.txts)} {time.time() - t0:.0f}s", flush=True)
    w.flush()
    pq.write_table(pa.Table.from_pylist(tasks), out / "locb_tasks.parquet", compression="zstd")
    cat = {r["instance_id"]: r["category"] for r in rows}
    ent = pq.read_table(out / "locb_ent_000.parquet", columns=["snap", "uid"]).to_pydict()
    by_snap = defaultdict(set)
    for s, u in zip(ent["snap"], ent["uid"]):
        by_snap[s].add(u)
    raw_gold_in = sum(all(g in by_snap[t["snap"]] for g in t["gold"]) for t in tasks)
    exact = sum(set(t["ours"]) == set(t["gold"]) for t in tasks)
    del ent, by_snap
    st = _convert(tasks, [out / "locb_ent_000.parquet"], [out / "locb_txt_000.parquet"], out, "locb", ["gold"],
                  eval_mode=True)
    pt = pq.read_table(out / "locb_tasks_paper.parquet").to_pylist()
    pe = pq.read_table(out / "locb_ent_paper_000.parquet", columns=["snap", "uid"]).to_pydict()
    ps = defaultdict(set)
    for s, u in zip(pe["snap"], pe["uid"]):
        ps[s].add(u)
    unreach = {t["instance_id"]: [g for g in t["gold"] if g not in ps[t["snap"]]] for t in pt}
    rep = {"dataset": LOCB, "revision": LOCB_REV, "instances_in_dataset": len(rows),
           "instances_built": len(tasks), "missing_instances": sorted(set(cat) - {t["instance_id"] for t in tasks}),
           "failures": dict(fails), "errors": errs[:50],
           "raw_gold_all_in_corpus": raw_gold_in, "label_agreement_exact_set": exact,
           "paper_format": st, "paper_tasks": len(pt),
           "paper_instances_with_unreachable_gold": sum(bool(v) for v in unreach.values()),
           "paper_unreachable_gold_sample": {k: v for k, v in list(unreach.items()) if v}.__repr__()[:3000],
           "paper_units_per_task_mean": sum(len(ps[t["snap"]]) for t in pt) / max(1, len(pt)),
           "category_counts": dict(Counter(cat[t["instance_id"]] for t in pt)),
           "sktime_tasks": sorted(t["instance_id"] for t in pt if t["repo"].lower() == "sktime/sktime"),
           "wall_s": time.time() - t0}
    (out / "report_locb.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "paper_unreachable_gold_sample"}, indent=1))
    if not a.limit_repos:
        assert len(tasks) == 560, f"only {len(tasks)} of 560 LocBench instances built"
    log("done")


if __name__ == "__main__":
    sys.exit(main())
