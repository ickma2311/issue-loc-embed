"""GitHub fork / parent / rename check of every training repo against every
evaluation repo (SWE-bench test+dev, SWE-bench Lite, LocBench V0.1/V0.2/V1). Read-only `gh api` calls.

A training repo and an evaluation repo are flagged when they are the same repository after GitHub
redirects, when one is the fork parent/source of the other, or when both sit in the same fork network
(same source root). Usage: python fork_check.py <train_pointers.parquet> <eval_repos.txt> <out.json>
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq


def info(repo):
    for _ in range(3):
        r = subprocess.run(["gh", "api", f"repos/{repo}"], capture_output=True, text=True)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            return {"requested": repo, "full_name": d["full_name"], "fork": d["fork"],
                    "parent": (d.get("parent") or {}).get("full_name"), "source": (d.get("source") or {}).get("full_name"),
                    "created_at": d.get("created_at")}
        if "Not Found" in r.stderr or "404" in r.stderr:
            return {"requested": repo, "error": "not_found"}
    return {"requested": repo, "error": r.stderr.strip()[:200]}


def main(ptr, evf, outf):
    train = sorted({r["repo"] for r in pq.read_table(ptr, columns=["repo"]).to_pylist()})
    ev = sorted({x.strip() for x in open(evf) if x.strip()})
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(info, train + ev))
    by = {r["requested"]: r for r in res}
    low = lambda x: x.lower() if x else None  # noqa: E731
    root = lambda r: low(r.get("source") or r.get("full_name") or r["requested"])  # noqa: E731
    ev_names = {low(e) for e in ev} | {low(by[e].get("full_name")) for e in ev if by[e].get("full_name")}
    ev_roots = {root(by[e]): e for e in ev}
    flags = []
    for t in train:
        r = by[t]
        why = []
        if low(r.get("full_name")) in ev_names:
            why.append("same repository after redirect")
        if low(r.get("parent")) in ev_names or low(r.get("source")) in ev_names:
            why.append("fork of an evaluation repo")
        if root(r) in ev_roots:
            why.append(f"same fork network as {ev_roots[root(r)]}")
        for e in ev:
            if low(by[e].get("parent")) in {low(t), low(r.get("full_name"))} or low(by[e].get("source")) in {low(t), low(r.get("full_name"))}:
                why.append(f"evaluation repo {e} is a fork of it")
        if why:
            flags.append({"train_repo": t, **r, "why": sorted(set(why))})
    out = {"n_train_repos": len(train), "n_eval_repos": len(ev),
           "train_errors": [by[t] for t in train if "error" in by[t]], "eval_errors": [by[e] for e in ev if "error" in by[e]],
           "train_forks": sum(1 for t in train if by[t].get("fork")),
           "train_renamed": [{"requested": t, "now": by[t]["full_name"]} for t in train
                             if by[t].get("full_name") and low(by[t]["full_name"]) != low(t)],
           "flags": flags, "repos": res}
    json.dump(out, open(outf, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("repos", "train_renamed")}, indent=1)[:4000])
    print("renamed training repos:", len(out["train_renamed"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
