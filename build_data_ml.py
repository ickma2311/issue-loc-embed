"""Build the multi-language training pool in the paper text format (Kaggle CPU).

Input: SWE-rebench V2 at a pinned revision and a task list (data/tasks_ml_v02.json: instance ids chosen by
scripts/select_ml.py, with the leakage check). For every repository: shallow-fetch the base commits, list the source files of the task language
(test files left out by the CoRNStack rule), parse units with tree-sitter (issueloc.units_ml), build the v0.1 pool
(patched files first, then whole files in hash(repo, path) order until >= POOL_CAP units), and label positives with
issueloc.patch.modified_units.
Outputs ($OUT_DIR/data): train_tasks_ml.parquet, train_ent_ml_XXX.parquet (snap, uid, kind, h),
train_txt_ml_XXX.parquet (h, path="", text), report_ml.json.
Usage: python build_data_ml.py [--limit-repos N] [--workers W] [--deadline-h H]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from issueloc import repo as gitrepo
from issueloc.paperfmt import h64, is_test_cornstack
from issueloc.patch import group_by_path, modified_units, parse_diff
from issueloc.units_ml import EXT, parse_units

V2_URL = ("https://huggingface.co/datasets/nebius/SWE-rebench-V2/resolve/"
          "10483de0f50fe5da545942705a76c6150171af7f/data/train-00000-of-00001.parquet")
POOL_CAP = 1000
MAX_POS = 20
WORK = Path(os.environ.get("TMP_WORK", "/tmp/build_ml_work"))


def log(msg):
    print(f"[build_ml] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def list_src(d: Path, sha: str, exts, repo: str) -> list[tuple[str, str]]:
    out = gitrepo._git(d, "ls-tree", "-r", "-z", sha)
    res = []
    for ent in out.split(b"\0"):
        if not ent:
            continue
        meta, path = ent.split(b"\t", 1)
        mode, typ, obj = meta.split(b" ")
        p = path.decode("utf-8", errors="replace")
        if typ == b"blob" and mode != b"120000" and p.endswith(exts) and not is_test_cornstack(p, repo):
            res.append((p, obj.decode()))
    return res


def process_repo(job: dict) -> dict:
    repo, lang, tasks = job["repo"], job["lang"], job["tasks"]
    exts = EXT[lang]
    t0 = time.time()
    res = {"repo": repo, "lang": lang, "tasks": [], "ents": [], "txts": {}, "fail": Counter()}
    try:
        d, got = gitrepo.fetch(repo, sorted({t["base_commit"] for t in tasks}), WORK)
    except Exception as e:
        res["fail"]["fetch_error"] += len(tasks); res["err"] = str(e)[:300]
        return res
    got = set(got)
    cache: dict[str, list | None] = {}   # blob sha -> [(Unit with real path, text)] or None (per path, see key)
    rng = random.Random(h64(repo))
    try:
        by_sha = defaultdict(list)
        for t in tasks:
            by_sha[t["base_commit"]].append(t)
        for sha, ts in by_sha.items():
            if sha not in got:
                res["fail"]["sha_missing"] += len(ts); continue
            files = list_src(d, sha, exts, repo)
            if not files:
                res["fail"]["no_src_files"] += len(ts); continue
            path2obj = dict(files)
            patched = set()
            for t in ts:
                patched |= {p for p in parse_diff(t["patch"]) if p in path2obj}
            order = sorted(path2obj, key=lambda p: h64(repo + "\0" + p))
            chosen = sorted(patched) + [p for p in order if p not in patched]
            pool, n_units_parsed, n_files_parsed, n_units_free, n_files_free = [], 0, 0, 0, 0
            for i, p in enumerate(chosen):
                key = path2obj[p] + "\0" + p  # units carry the path in their text, so cache per (blob, path)
                if key not in cache:
                    batch = []
                    for q in chosen[i:]:
                        if path2obj[q] + "\0" + q not in cache:
                            batch.append((path2obj[q], q))
                            if len(batch) >= 64:
                                break
                    texts = gitrepo.read_blobs(d, sorted({b for b, _ in batch}))
                    for b, q in batch:
                        src = texts.get(b)
                        cache[b + "\0" + q] = parse_units(src, q, lang) if src is not None else None
                parsed = cache.get(key)
                if parsed is not None:
                    n_files_parsed += 1; n_units_parsed += len(parsed)
                    if p not in patched:
                        n_files_free += 1; n_units_free += len(parsed)
                    pool.extend(parsed)
                if i >= len(patched) - 1 and len(pool) >= POOL_CAP:
                    break
            if n_files_free:
                rest = len(files) - (n_files_parsed - n_files_free)
                full_est = int(round((n_units_parsed - n_units_free) + rest * n_units_free / n_files_free))
            else:
                full_est = len(pool)
            full_est = max(full_est, len(pool))
            snap = f"{repo}@{sha}"
            ent_rows = []
            for u, txt in pool:
                h = h64(txt)
                res["txts"].setdefault(h, txt)
                ent_rows.append((snap, u.uid, u.kind, h))
            res["ents"].extend(ent_rows)
            ubp = group_by_path([u for u, _ in pool])
            pool_ids = {r[1] for r in ent_rows}
            for t in ts:
                ours = [x for x in modified_units(t["patch"], ubp) if x in pool_ids]
                if not ours:
                    res["fail"]["no_positive_unit"] += 1; continue
                pos = ours if len(ours) <= MAX_POS else sorted(rng.sample(ours, MAX_POS))
                res["tasks"].append({"instance_id": t["instance_id"], "repo": repo, "language": lang, "snap": snap,
                                     "query": t["query"], "positives": pos, "all_positives": ours,
                                     "excl_positives": ours, "n_pos_raw": len(ours), "pool_n": len(ent_rows),
                                     "full_est": full_est})
    except Exception as e:
        res["fail"]["repo_exception"] += 1; res["err"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res["secs"] = time.time() - t0
    return res


class Shards:
    def __init__(self, out: Path, every: int = 200000):
        self.out, self.every, self.n, self.ents, self.txts, self.seen = out, every, 0, [], {}, set()

    def add(self, res):
        self.ents.extend(res["ents"])
        for h, t in res["txts"].items():
            if h not in self.seen:
                self.seen.add(h); self.txts[h] = t
        if len(self.ents) >= self.every:
            self.flush()

    def flush(self):
        if not self.ents and not self.txts:
            return
        s, u, k, h = zip(*self.ents) if self.ents else ([], [], [], [])
        pq.write_table(pa.table({"snap": list(s), "uid": list(u), "kind": list(k), "h": pa.array(list(h), pa.int64())}),
                       self.out / f"train_ent_ml_{self.n:03d}.parquet", compression="zstd")
        hs = list(self.txts)
        pq.write_table(pa.table({"h": pa.array(hs, pa.int64()), "path": pa.array([""] * len(hs), pa.string()),
                                 "text": pa.array([self.txts[x] for x in hs], pa.large_string())}),
                       self.out / f"train_txt_ml_{self.n:03d}.parquet", compression="zstd")
        self.n += 1; self.ents, self.txts = [], {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks_ml.json")
    ap.add_argument("--limit-repos", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--deadline-h", type=float, default=10.5)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(os.environ.get("OUT_DIR", "out")) / "data"
    out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    want = {i: L for L, ids in json.load(open(a.tasks)).items() for i in ids}
    v2f = WORK / "v2.parquet"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(V2_URL, v2f); break
        except Exception as e:
            log(f"retry V2 download: {e}"); time.sleep(20 * (attempt + 1))
    rows = [r for r in pq.read_table(v2f, columns=["instance_id", "repo", "language", "base_commit", "problem_statement",
                                                   "patch"]).to_pylist() if r["instance_id"] in want]
    assert len(rows) == len(want), (len(rows), len(want))
    by_repo = defaultdict(list)
    for r in rows:
        assert r["language"] == want[r["instance_id"]]
        by_repo[(r["repo"], r["language"])].append({"instance_id": r["instance_id"], "base_commit": r["base_commit"],
                                                    "patch": r["patch"] or "", "query": r["problem_statement"] or ""})
    jobs = [{"repo": k[0], "lang": k[1], "tasks": v} for k, v in sorted(by_repo.items())]
    if a.limit_repos:
        jobs = random.Random(0).sample(jobs, min(a.limit_repos, len(jobs)))
    log(f"{len(rows)} tasks, {len(jobs)} repositories")
    shards, tasks, fail, errs, done = Shards(out), [], Counter(), [], 0
    deadline = t_start + a.deadline_h * 3600
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_repo, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                res = f.result()
            except Exception as e:
                fail["worker_error"] += len(j["tasks"]); errs.append((j["repo"], str(e)[:200])); continue
            tasks.extend(res["tasks"]); fail.update(res["fail"]); shards.add(res)
            if res.get("err"):
                errs.append((j["repo"], res["err"]))
            done += 1
            if done % 50 == 0:
                print(f"[progress] build-ml-repos {done}/{len(jobs)} {time.time() - t_start:.0f}s tasks {len(tasks)}", flush=True)
            if time.time() > deadline:
                log("deadline reached: stop submitting, cancel the rest")
                for g in futs:
                    g.cancel()
                fail["deadline_unprocessed"] += sum(len(futs[g]["tasks"]) for g in futs if not g.done())
                break
    shards.flush()
    tasks.sort(key=lambda t: t["instance_id"])
    pq.write_table(pa.Table.from_pylist(tasks), out / "train_tasks_ml.parquet", compression="zstd")
    rep = {"tasks_in": len(rows), "tasks_out": len(tasks), "repos": len(jobs), "per_language": dict(Counter(t["language"] for t in tasks)),
           "positives": sum(len(t["positives"]) for t in tasks), "fail": dict(fail), "errors_sample": errs[:50],
           "n_errors": len(errs), "unique_texts": len(shards.seen), "hours": (time.time() - t_start) / 3600,
           "mean_pool": sum(t["pool_n"] for t in tasks) / max(1, len(tasks))}
    (out / "report_ml.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps({k: v for k, v in rep.items() if k != "errors_sample"}))


if __name__ == "__main__":
    main()
