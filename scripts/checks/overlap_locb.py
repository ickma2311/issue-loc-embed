"""Overlap of the training data with LocBench V1 before any LocBench evaluation.

Checks: repository (exact, after GitHub redirects / forks via fork_check.json, shared project name or owner),
instance id and base commit, issue text (8-word-shingle containment, links to LocBench PRs), gold functions
(same path:qualname among training positives; identical function bodies of >= 200 characters between LocBench
gold units and training pool texts), and time. CPU only.
Usage: python overlap_locb.py <train_pointers.parquet> <hfdl_dir> <evidence_paper_data_dir> <locb_data_dir>
                              <fork_check.json> <out.json>
"""
import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

LOCB_FILE = "czlll__Loc-Bench_V1__test__0000.parquet"


def sh(t, n=8):
    w = re.findall(r"\w+", (t or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def body(txt):
    b = txt.split("\n", 1)[1] if "\n" in txt else txt
    return re.sub(r"\s+", " ", b).strip()


def H(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=12).digest()


def main(ptr_f, dl, paper_dir, locb_dir, fork_f, out_f):
    ptr = pq.read_table(ptr_f).to_pylist()
    loc = pq.read_table(Path(dl) / LOCB_FILE).to_pylist()
    loc_repos = {r["repo"].lower() for r in loc}
    tr_repos = {r["repo"].lower() for r in ptr}
    res = {"locbench_instances": len(loc), "locbench_repos": len(loc_repos), "train_tasks": len(ptr),
           "train_repos": len(tr_repos)}
    # 1) repositories
    res["repo_exact_overlap"] = sorted(tr_repos & loc_repos)
    names = collections.defaultdict(list)
    for r in loc_repos:
        names[r.split("/")[1]].append(r)
    res["train_repos_sharing_project_name"] = sorted(f"{t} ~ {names[t.split('/')[1]]}" for t in tr_repos if t.split("/")[1] in names)
    owners = {r.split("/")[0] for r in loc_repos}
    res["train_repos_sharing_owner"] = sorted(t for t in tr_repos if t.split("/")[0] in owners)
    fk = json.loads(Path(fork_f).read_text())
    loc_low = {r.lower() for r in loc_repos}
    res["fork_flags_involving_locbench"] = [f for f in fk["flags"]
                                            if any(x in loc_low for x in [(f.get("full_name") or "").lower(),
                                                                          (f.get("parent") or "").lower(),
                                                                          (f.get("source") or "").lower()])
                                            or any(w.split()[-1].lower() in loc_low for w in f["why"] if " " in w)]
    # 2) instances / commits
    res["instance_id_overlap"] = sorted({r["instance_id"] for r in ptr} & {r["instance_id"] for r in loc})
    res["base_commit_overlap"] = sorted({r["base_commit"] for r in ptr} & {r["base_commit"] for r in loc})
    # 3) issue text
    rows = {}
    for f in sorted(glob.glob(str(Path(dl) / "nebius__SWE-rebench*.parquet"))):
        for r in pq.read_table(f, columns=["instance_id", "problem_statement", "created_at"]).to_pylist():
            rows.setdefault(r["instance_id"], r)
    pids = [r["instance_id"] for r in ptr]
    assert all(i in rows for i in pids), "pointer ids missing from SWE-rebench download"
    L = {r["instance_id"]: sh(r["problem_statement"]) for r in loc}
    inv = collections.defaultdict(set)
    for k, s in L.items():
        for x in s:
            inv[x].add(k)
    best = collections.defaultdict(lambda: (0.0, None))
    for i in pids:
        c = collections.Counter()
        for x in sh(rows[i]["problem_statement"]):
            for k in inv.get(x, ()):
                c[k] += 1
        for k, v in c.items():
            fr = v / max(1, len(L[k]))
            if fr > best[k][0]:
                best[k] = (fr, i)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:10]
    res["issue_containment_top10"] = [(k, round(f, 3), i) for k, (f, i) in top]
    res["issue_containment_ge_0.5"] = sum(f >= 0.5 for f, _ in best.values())
    res["issue_containment_ge_0.3"] = sum(f >= 0.3 for f, _ in best.values())
    pr = collections.defaultdict(set)
    for r in loc:
        m = re.search(r"-(\d+)$", r["instance_id"])
        pr[r["repo"].lower()].add(m.group(1))
    xref = []
    for i in pids:
        for m in re.finditer(r"github\.com/([\w.-]+/[\w.-]+)/(?:issues|pull)/(\d+)", rows[i]["problem_statement"] or ""):
            if m.group(2) in pr.get(m.group(1).lower(), ()):
                xref.append((i, m.group(0)))
    res["train_issues_linking_a_locbench_pr"] = xref
    # 4) gold functions: identical path:qualname among training positives
    gold = collections.defaultdict(list)
    for r in loc:
        for g in r["edit_functions"]:
            gold[g].append(r["instance_id"])
    pos = collections.defaultdict(list)
    for r in ptr:
        for p in r["positives"]:
            pos[p].append(r["instance_id"])
    ex = sorted(set(gold) & set(pos))
    res["gold_path_qualname_equal_to_training_positive"] = [(g, gold[g][:2], pos[g][:2]) for g in ex][:50]
    res["gold_path_qualname_equal_count"] = len(ex)
    # 5) identical bodies (>= 200 chars) between LocBench gold units and training pool texts
    lt = pq.read_table(Path(locb_dir) / "locb_tasks_paper.parquet").to_pylist()
    le = pq.read_table(Path(locb_dir) / "locb_ent_paper_000.parquet", columns=["snap", "uid", "h"]).to_pydict()
    key2h = {(s, u): h for s, u, h in zip(le["snap"], le["uid"], le["h"])}
    need = {key2h[(t["snap"], g)]: (t["instance_id"], g) for t in lt for g in t["gold"] if (t["snap"], g) in key2h}
    gb = {}
    for b in pq.ParquetFile(Path(locb_dir) / "locb_txt_paper_000.parquet").iter_batches(batch_size=200000, columns=["h", "text"]):
        for h, x in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            if h in need:
                bd = body(x)
                if len(bd) >= 200:
                    gb.setdefault(H(bd), []).append(need[h])
    te = pq.read_table(Path(paper_dir) / "train_ent_paper_000.parquet", columns=["snap", "uid", "h"]).to_pydict()
    tt = pq.read_table(Path(paper_dir) / "train_tasks_paper.parquet", columns=["snap", "positives"]).to_pylist()
    tpos = {(t["snap"], p) for t in tt for p in t["positives"]}
    h_pos = collections.defaultdict(bool)
    h_where = {}
    for s, u, h in zip(te["snap"], te["uid"], te["h"]):
        h_pos[h] |= (s, u) in tpos
        h_where.setdefault(h, f"{s.split('@')[0]}:{u}")
    hits = []
    for b in pq.ParquetFile(Path(paper_dir) / "train_txt_paper_000.parquet").iter_batches(batch_size=100000, columns=["h", "text"]):
        for h, x in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            k = H(body(x))
            if k in gb:
                hits.append({"train_unit": h_where.get(h), "train_positive": bool(h_pos[h]), "locbench_gold": gb[k][:2]})
    res["gold_units_ge200_chars"] = sum(len(v) for v in gb.values())
    res["gold_bodies_identical_in_training_pool"] = len(hits)
    res["gold_bodies_identical_to_training_positive"] = sum(x["train_positive"] for x in hits)
    res["gold_body_hits"] = hits[:60]
    res["gold_body_hit_locbench_tasks"] = sorted({g[0] for x in hits for g in x["locbench_gold"]})
    # 6) time
    lc = sorted(r["created_at"] for r in loc)  # ms since epoch
    import datetime as dt
    tr_ca = sorted(str(rows[i]["created_at"]) for i in pids if rows[i].get("created_at"))
    latest = dt.datetime.utcfromtimestamp(lc[-1] / 1000).strftime("%Y-%m-%d")
    res["locbench_created_range"] = [dt.datetime.utcfromtimestamp(lc[0] / 1000).strftime("%Y-%m-%d"), latest]
    res["train_created_range"] = [tr_ca[0][:10], tr_ca[-1][:10]]
    res["train_tasks_before_latest_locbench_issue"] = sum(c[:10] <= latest for c in tr_ca)
    res["sktime_locbench_tasks"] = sorted(r["instance_id"] for r in loc if r["repo"].lower() == "sktime/sktime")
    Path(out_f).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("gold_body_hits",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(*sys.argv[1:7])
