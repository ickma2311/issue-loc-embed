"""Data build (Kaggle CPU): the SweLoc-style training pool and the SWE-Bench-Lite eval corpus.

Outputs ($OUT_DIR/data):
  train_tasks.parquet    instance_id, repo, snap, query, positives, n_pos_raw, pool_n, full_est
  train_ent_XXX.parquet  snap, uid, kind, h            (one row per pool unit per snapshot)
  train_txt_XXX.parquet  h, path, text                 (unique unit texts)
  eval_tasks.parquet     instance_id, repo, snap, query, gold, ours
  eval_ent.parquet / eval_txt.parquet                 (full corpus, every .py file)
  report.json            counts, failures, label agreement with LocAgent gold, train/eval overlap

Usage: python build_data.py [--limit-repos N] [--deadline-h H] [--workers W]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from issueloc import repo as gitrepo
from issueloc.revisions import parquet_url
from issueloc.patch import group_by_path, modified_units, touched_py_files
from issueloc.tasks import s_, select
from issueloc.units import Unit, parse_units, unit_text

POOL_CAP = 1000          # units per training snapshot (file-level sampling, patched files always in)
MAX_POS = 20             # positives per task
MAX_TASKS_PER_REPO = 100
WORK = Path(os.environ.get("TMP_WORK", "/tmp/issueloc_build"))


def log(msg):
    print(f"[build] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def read_hf(ds, split, parts, cols):
    tabs = []
    for p in parts:
        for attempt in range(4):
            try:
                with fsspec.open(parquet_url(ds, "default", split, p), "rb", block_size=8 << 20) as f:
                    tabs.append(pq.read_table(f, columns=cols))
                break
            except Exception as e:  # transient HF errors
                log(f"retry {ds} {split} {p}: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"cannot read {ds}")
    return pa.concat_tables(tabs, promote_options="default").to_pylist()


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little", signed=True)


def norm_issue(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()[:300]


def process_repo(job: dict) -> dict:
    """job: {repo, split, tasks:[{instance_id, base_commit, patch, query, gold?}]}"""
    repo, split, tasks = job["repo"], job["split"], job["tasks"]
    t0 = time.time()
    res = {"repo": repo, "split": split, "tasks": [], "ents": [], "txts": {}, "fail": Counter()}
    try:
        d, got = gitrepo.fetch(repo, sorted({t["base_commit"] for t in tasks}), WORK)
    except Exception as e:
        res["fail"]["fetch_error"] += len(tasks)
        res["err"] = str(e)[:300]
        return res
    got = set(got)
    blob_cache: dict[str, list | None] = {}   # blob sha -> [(Unit, text)] or None
    rng = random.Random(h64(repo))
    try:
        by_sha = defaultdict(list)
        for t in tasks:
            by_sha[t["base_commit"]].append(t)
        for sha, ts in by_sha.items():
            if sha not in got:
                res["fail"]["sha_missing"] += len(ts)
                continue
            files = gitrepo.list_py(d, sha)
            if not files:
                res["fail"]["no_py_files"] += len(ts)
                continue
            path2obj = dict(files)
            patched = set()
            for t in ts:
                patched |= {p for p in touched_py_files(t["patch"]) if p in path2obj}
            if split == "train":
                order = sorted(path2obj, key=lambda p: h64(repo + "\0" + p))
                chosen = [p for p in sorted(patched)] + [p for p in order if p not in patched]
            else:
                chosen = sorted(path2obj)
            # read + parse lazily in chosen order (64 blobs per git call) until the pool cap is reached
            pool_units, n_units_parsed, n_files_parsed = [], 0, 0
            n_units_free, n_files_free = 0, 0   # non-patched files only, for the full-size estimate
            n_forced = len(patched)
            for i, p in enumerate(chosen):
                obj = path2obj[p]
                if obj not in blob_cache:
                    batch = []
                    for q in chosen[i:]:
                        o = path2obj[q]
                        if o not in blob_cache and o not in batch:
                            batch.append(o)
                            if len(batch) >= 64:
                                break
                    texts = gitrepo.read_blobs(d, batch)
                    for o in batch:
                        src = texts.get(o)
                        units = parse_units(src, "\0") if src is not None else None
                        if units is None:
                            blob_cache[o] = None
                            continue
                        lines = src.splitlines(keepends=True)
                        blob_cache[o] = [(u, unit_text(lines, u)) for u in units]
                parsed = blob_cache.get(obj)
                if parsed is not None:
                    n_files_parsed += 1
                    n_units_parsed += len(parsed)
                    if p not in patched:
                        n_files_free += 1
                        n_units_free += len(parsed)
                    pool_units.extend((p, u, txt) for u, txt in parsed)
                if split == "train" and i >= n_forced - 1 and len(pool_units) >= POOL_CAP:
                    break
            if split == "train":
                if n_files_free:
                    # patched files are counted exactly; the rest of the repo by the sampled-file mean
                    n_patched_units = n_units_parsed - n_units_free
                    rest = len(files) - (n_files_parsed - n_files_free)
                    full_est = int(round(n_patched_units + rest * n_units_free / n_files_free))
                else:
                    full_est = len(pool_units)
                full_est = max(full_est, len(pool_units))
            else:
                full_est = len(pool_units)
            snap = f"{repo}@{sha}"
            # unit ids with the real path
            units_by_path = defaultdict(list)
            ent_rows = []
            for p, u, txt in pool_units:
                uid = p + u.uid[1:]  # u.uid == "\0:qualname"
                uu = Unit(uid, u.kind, u.start, u.end, u.indent, None)
                units_by_path[p].append(uu)
                h = h64(p + "\0" + txt)
                res["txts"].setdefault(h, (p, txt))
                ent_rows.append((snap, uid, u.kind, h))
            res["ents"].extend(ent_rows)
            pool_ids = {r[1] for r in ent_rows}
            for t in ts:
                ours = [x for x in modified_units(t["patch"], units_by_path) if x in pool_ids]
                row = {"instance_id": t["instance_id"], "repo": repo, "snap": snap, "query": t["query"],
                       "pool_n": len(ent_rows), "full_est": full_est}
                if split == "train":
                    if not ours:
                        res["fail"]["no_positive_unit"] += 1
                        continue
                    pos = ours if len(ours) <= MAX_POS else sorted(rng.sample(ours, MAX_POS))
                    row.update(positives=pos, all_positives=ours, n_pos_raw=len(ours))
                else:
                    row.update(gold=list(t["gold"]), ours=ours)
                res["tasks"].append(row)
    except Exception as e:
        res["fail"]["repo_exception"] += 1
        res["err"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res["secs"] = time.time() - t0
    return res


class ShardWriter:
    def __init__(self, out: Path, prefix: str, every: int):
        self.out, self.prefix, self.every = out, prefix, every
        self.ents, self.txts, self.n = [], {}, 0

    def add(self, res):
        self.ents.extend(res["ents"])
        self.txts.update(res["txts"])
        if len(self.ents) >= self.every:
            self.flush()

    def flush(self):
        if not self.ents:
            return
        snap, uid, kind, h = zip(*self.ents)
        pq.write_table(pa.table({"snap": snap, "uid": uid, "kind": kind, "h": pa.array(h, pa.int64())}),
                       self.out / f"{self.prefix}_ent_{self.n:03d}.parquet", compression="zstd")
        hs = list(self.txts)
        pq.write_table(pa.table({"h": pa.array(hs, pa.int64()), "path": [self.txts[x][0] for x in hs],
                                 "text": [self.txts[x][1] for x in hs]}),
                       self.out / f"{self.prefix}_txt_{self.n:03d}.parquet", compression="zstd")
        self.ents, self.txts, self.n = [], {}, self.n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-repos", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0, help="eval instances (smoke)")
    ap.add_argument("--eval-repo", default=None, help="only this eval repo (smoke)")
    ap.add_argument("--deadline-h", type=float, default=9.5)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(os.environ.get("OUT_DIR", "out")) / "data"
    out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    log("loading task tables from HF")
    ev_repos = set()
    for ds, splits in [("princeton-nlp/SWE-bench", ["test", "dev"]), ("princeton-nlp/SWE-bench_Lite", ["test", "dev"]),
                       ("czlll/Loc-Bench_V1", ["test"]), ("czlll/Loc-Bench_V0.2", ["test"]), ("czlll/Loc-Bench_V0.1", ["test"])]:
        for s in splits:
            ev_repos |= {r["repo"].lower() for r in read_hf(ds, s, ["0000"], ["repo"])}
    log(f"evaluation repos: {len(ev_repos)}")
    cols = ["instance_id", "repo", "base_commit", "problem_statement", "patch"]
    v1 = read_hf("nebius/SWE-rebench", "test", ["0000", "0001"], cols + ["license_name"])
    for r in v1:
        r["license"] = r.pop("license_name")
    v2 = [r for r in read_hf("nebius/SWE-rebench-V2", "train", ["0000"], cols + ["license", "language"])
          if r["language"] == "python"]
    kept, stats = select(v1, v2, ev_repos)
    log(f"selected {len(kept)} tasks (expected 20574) {dict(stats)}")
    assert len(kept) == 20574, "task selection differs from the expected 20,574 (HF parquet parts changed?)"
    evrows = read_hf("czlll/SWE-bench_Lite", "test", ["0000"], cols + ["edit_functions"])
    log(f"eval instances: {len(evrows)}")
    assert len(evrows) == 274 and all(len(r["edit_functions"]) for r in evrows)
    ev_names = {r.split("/")[1].lower() for r in ev_repos}
    fork_like = sorted({r["repo"] for r in kept if r["repo"].split("/")[1].lower() in ev_names})
    log(f"training repos sharing a project name with an eval repo (dropped before training: paperfmt and gpu_run.load_train): {fork_like[:30]}")

    # train/eval issue overlap (exact normalised text)
    ev_norm = {norm_issue(s_(r["problem_statement"])) for r in evrows}
    overlap = [r["instance_id"] for r in kept if norm_issue(s_(r["problem_statement"])) in ev_norm]
    kept = [r for r in kept if norm_issue(s_(r["problem_statement"])) not in ev_norm]

    rng = random.Random(0)
    by_repo = defaultdict(list)
    for r in kept:
        by_repo[r["repo"]].append({"instance_id": r["instance_id"], "base_commit": r["base_commit"],
                                   "patch": s_(r["patch"]), "query": s_(r["problem_statement"])})
    capped = 0
    for k, v in by_repo.items():
        if len(v) > MAX_TASKS_PER_REPO:
            capped += len(v) - MAX_TASKS_PER_REPO
            by_repo[k] = rng.sample(v, MAX_TASKS_PER_REPO)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    if a.limit_repos:
        repos = repos[:a.limit_repos]
    ev_by_repo = defaultdict(list)
    if a.eval_repo:
        evrows = [r for r in evrows if r["repo"] == a.eval_repo]
    for r in evrows[: a.limit_eval or None]:
        ev_by_repo[r["repo"]].append({"instance_id": r["instance_id"], "base_commit": r["base_commit"],
                                      "patch": s_(r["patch"]), "query": s_(r["problem_statement"]),
                                      "gold": list(r["edit_functions"])})
    jobs = [{"repo": k, "split": "eval", "tasks": v} for k, v in ev_by_repo.items()]
    jobs += [{"repo": k, "split": "train", "tasks": by_repo[k]} for k in repos]
    log(f"jobs: {len(ev_by_repo)} eval repos, {len(repos)} train repos ({sum(len(by_repo[k]) for k in repos)} tasks)")

    tw = ShardWriter(out, "train", 2_000_000)
    ew = ShardWriter(out, "eval", 10**12)
    train_tasks, eval_tasks, fails, errs, secs = [], [], Counter(), [], []
    deadline = t_start + a.deadline_h * 3600
    done = 0
    broken = False
    ex = ProcessPoolExecutor(a.workers)
    try:
        pending = {}
        it = iter(jobs)
        def submit_next():
            j = next(it, None)
            if j is not None and time.time() < deadline and not broken:
                pending[ex.submit(process_repo, j)] = j
        for _ in range(a.workers * 2):
            submit_next()
        while pending:
            for fut in as_completed(list(pending)):
                j = pending.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool as e:
                    broken = True
                    fails["worker_crash"] += len(j["tasks"])
                    errs.append((j["repo"], "BrokenProcessPool " + repr(e)[:150]))
                    break
                except Exception as e:
                    fails["worker_crash"] += len(j["tasks"])
                    errs.append((j["repo"], repr(e)[:200]))
                    submit_next()
                    break
                fails.update(res["fail"])
                if res.get("err"):
                    errs.append((res["repo"], res["err"]))
                secs.append(res.get("secs", 0))
                if res["split"] == "eval":
                    eval_tasks += res["tasks"]
                    ew.add(res)
                else:
                    train_tasks += res["tasks"]
                    tw.add(res)
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    el = time.time() - t_start
                    print(f"[progress] {done}/{len(jobs)} repos  train_tasks={len(train_tasks)} "
                          f"eval_tasks={len(eval_tasks)} {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s",
                          flush=True)
                submit_next()
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    tw.flush()
    ew.flush()
    pq.write_table(pa.Table.from_pylist(train_tasks), out / "train_tasks.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(eval_tasks), out / "eval_tasks.parquet", compression="zstd")

    # agreement of our patch->unit labels with LocAgent gold (diagnostic only)
    exact = sum(set(t["ours"]) == set(t["gold"]) for t in eval_tasks)
    g_tot = sum(len(t["gold"]) for t in eval_tasks)
    g_hit = sum(len(set(t["ours"]) & set(t["gold"])) for t in eval_tasks)
    o_tot = sum(len(t["ours"]) for t in eval_tasks)
    gold_in_corpus = 0
    if eval_tasks:
        ent = pq.read_table(out / "eval_ent_000.parquet", columns=["snap", "uid"]).to_pydict()
        by_snap = defaultdict(set)
        for s, u in zip(ent["snap"], ent["uid"]):
            by_snap[s].add(u)
        gold_in_corpus = sum(all(g in by_snap[t["snap"]] for g in t["gold"]) for t in eval_tasks)
    rep = {
        "selected_tasks": len(kept) + len(overlap), "selection_stats": dict(stats),
        "train_eval_issue_overlap_dropped": overlap, "tasks_dropped_by_repo_cap": capped,
        "train_repos_processed": sum(1 for _ in secs) - len(ev_by_repo),
        "train_tasks_with_positives": len(train_tasks),
        "train_positives": sum(len(t["positives"]) for t in train_tasks),
        "train_pool_units_mean": (sum(t["pool_n"] for t in train_tasks) / max(1, len(train_tasks))),
        "eval_tasks": len(eval_tasks), "eval_gold_all_in_corpus": gold_in_corpus,
        "label_agreement": {"exact_set_match": exact, "gold_recall": g_hit / max(1, g_tot),
                            "ours_precision": g_hit / max(1, o_tot)},
        "failures": dict(fails), "errors_sample": errs[:50], "n_errors": len(errs),
        "deadline_hit": time.time() >= deadline, "pool_broken": broken, "wall_s": time.time() - t_start,
        "fork_like_train_repos": fork_like,
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("errors_sample", "train_eval_issue_overlap_dropped")}, indent=1))
    if not a.limit_eval and not a.eval_repo:
        assert len(eval_tasks) == 274, f"only {len(eval_tasks)} eval instances built"
    log("done")


if __name__ == "__main__":
    sys.exit(main())
