"""Pin the Hugging Face dataset revisions the data build read (2026-10-12) and check them.

The build read the auto-converted parquet branch (refs/convert/parquet) without a revision. For every dataset this
records the full commit ids of that branch and of main as of the build (no commit after it exists), downloads
the exact parquet parts at those commits, hashes them, re-runs the task selection (issueloc.tasks.select) and
compares the result with the build output files (train_tasks.parquet / eval_tasks.parquet).
Usage: python hf_revisions.py <build_data.py output dir> <out.json> <download dir>
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from issueloc.tasks import select  # noqa: E402

BUILD_TIME = "2026-10-12T23:45:32Z"
READ = {  # dataset -> (split dir, parts) exactly as build_data.read_hf read them
    "nebius/SWE-rebench": [("test", "0000"), ("test", "0001")],
    "nebius/SWE-rebench-V2": [("train", "0000")],
    "czlll/SWE-bench_Lite": [("test", "0000")],
    "princeton-nlp/SWE-bench": [("test", "0000"), ("dev", "0000")],
    "princeton-nlp/SWE-bench_Lite": [("test", "0000"), ("dev", "0000")],
    "czlll/Loc-Bench_V1": [("test", "0000")],
    "czlll/Loc-Bench_V0.2": [("test", "0000")],
    "czlll/Loc-Bench_V0.1": [("test", "0000")],
}


ALSO_READ = ["mteb/CodeSearchNetRetrieval"]  # read by csn_eval.py (auto-converted parquet branch), no selection check


def api(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def commit_at(ds, branch):
    cs = api(f"https://huggingface.co/api/datasets/{ds}/commits/{branch.replace('/', '%2F')}")
    before = [c for c in cs if c["date"] <= BUILD_TIME]
    return {"id": before[0]["id"], "date": before[0]["date"], "latest_now": cs[0]["id"],
            "changed_since_build": cs[0]["id"] != before[0]["id"]}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(ev_dir, outf, dl):
    dl = Path(dl); dl.mkdir(parents=True, exist_ok=True)
    out, files = {}, {}
    for ds, parts in READ.items():
        rev = {"parquet": commit_at(ds, "refs/convert/parquet"), "main": commit_at(ds, "main")}
        got = []
        for split, part in parts:
            url = f"https://huggingface.co/datasets/{ds}/resolve/{rev['parquet']['id']}/default/{split}/{part}.parquet"
            p = dl / f"{ds.replace('/', '__')}__{split}__{part}.parquet"
            if not p.exists():
                urllib.request.urlretrieve(url, p)
            files[(ds, split, part)] = p
            got.append({"split": split, "part": part, "url": url, "sha256": sha256(p), "rows": pq.read_metadata(p).num_rows})
        out[ds] = {**rev, "parts": got}
    for ds in ALSO_READ:
        out[ds] = {"parquet": commit_at(ds, "refs/convert/parquet"), "main": commit_at(ds, "main")}
    rd = lambda ds, split, part, cols: pq.read_table(files[(ds, split, part)], columns=cols).to_pylist()  # noqa: E731
    ev_repos = set()
    for ds in ("princeton-nlp/SWE-bench", "princeton-nlp/SWE-bench_Lite", "czlll/Loc-Bench_V1", "czlll/Loc-Bench_V0.2", "czlll/Loc-Bench_V0.1"):
        for split, part in READ[ds]:
            ev_repos |= {r["repo"].lower() for r in rd(ds, split, part, ["repo"])}
    cols = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
    v1 = rd("nebius/SWE-rebench", "test", "0000", cols + ["license_name"]) + rd("nebius/SWE-rebench", "test", "0001", cols + ["license_name"])
    for r in v1:
        r["license"] = r.pop("license_name")
    v2 = [r for r in rd("nebius/SWE-rebench-V2", "train", "0000", cols + ["license", "language"]) if r["language"] == "python"]
    kept, stats = select(v1, v2, ev_repos)
    sel = {r["instance_id"] for r in kept}
    built = pq.read_table(Path(ev_dir) / "train_tasks.parquet", columns=["instance_id"]).column(0).to_pylist()
    ev = pq.read_table(Path(ev_dir) / "eval_tasks.parquet", columns=["instance_id"]).column(0).to_pylist()
    lite = {r["instance_id"] for r in rd("czlll/SWE-bench_Lite", "test", "0000", ["instance_id"])}
    lite_dates = sorted(str(r["created_at"])[:10] for r in rd("princeton-nlp/SWE-bench_Lite", "test", "0000", ["instance_id", "created_at"])
                        if r["instance_id"] in lite)
    check = {"eval_repos": len(ev_repos), "eval_repos_expected": 229, "selected": len(kept), "selected_expected": 20574,
             "selection_stats": dict(stats), "build_train_tasks": len(built), "build_train_tasks_not_in_selection": len(set(built) - sel),
             "build_eval_tasks": len(ev), "build_eval_tasks_equal_czlll_lite": set(ev) == lite,
             "lite_274_created_range": [lite_dates[0], lite_dates[-1]],
             "build_output_sha256": {f: sha256(Path(ev_dir) / f) for f in ("train_tasks.parquet", "eval_tasks.parquet")}}
    check["pass"] = (check["eval_repos"] == 229 and check["selected"] == 20574
                     and check["build_train_tasks_not_in_selection"] == 0 and check["build_eval_tasks_equal_czlll_lite"])
    res = {"build_started": BUILD_TIME, "datasets": out, "check": check}
    Path(outf).write_text(json.dumps(res, indent=1))
    print(json.dumps(check, indent=1))
    print({ds: (v["parquet"]["id"][:12], v["parquet"]["changed_since_build"]) for ds, v in out.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
