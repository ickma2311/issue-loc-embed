"""Literal CoRNStack build (Kaggle CPU; diagnostic only, not used for any reported number): SWE-Bench-Lite function-localization eval set built exactly as CoRNStack
(Suresh et al., 2024) builds it, which SweRank (arXiv 2505.07849 §5.1) follows. Uses the vendored
Apache-2.0 CoRNStack code in third_party/cornstack (create_structure, parse_patch_full,
find_py_or_non_dict_with_path) with the same call pattern as cornstack/src/create/swebench.py::function:
  corpus = functions + direct class methods of non-test files, text = "<id>\\n<code>"
  gold   = functions/methods touched by the patch, excluding newly added ones
  instances without gold are dropped (paper: 274 of 300 remain)
Outputs to $OUT_DIR/data: eval_tasks_paper.parquet, eval_ent_paper_000.parquet, eval_txt_paper_000.parquet
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_data import read_hf
from issueloc.repo import ENV
from third_party.cornstack.get_patch_info import find_py_or_non_dict_with_path, parse_patch_full
from third_party.cornstack.get_repo_structure import create_structure, repo_to_top_folder

WORK = Path(os.environ.get("TMP_WORK", "/tmp/issueloc_cornstack"))


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little", signed=True)


def git(cwd, *args, **kw):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True, capture_output=True, **kw).stdout


def main():
    out = Path(os.environ.get("OUT_DIR", "out")) / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_hf("princeton-nlp/SWE-bench_Lite", "test", ["0000"],
                   ["instance_id", "repo", "base_commit", "problem_statement", "patch"])
    assert len(rows) == 300
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    tasks, ents, txts, dropped, t0 = [], [], {}, [], time.time()
    for repo, rs in sorted(by_repo.items()):
        top = repo_to_top_folder[repo]
        g = WORK / "git" / top
        shutil.rmtree(g, ignore_errors=True)
        g.mkdir(parents=True)
        git(g, "init", "-q")
        git(g, "remote", "add", "origin", f"https://github.com/{repo}.git")
        shas = sorted({r["base_commit"] for r in rs})
        for i in range(0, len(shas), 40):
            git(g, "fetch", "-q", "--depth", "1", "--no-tags", "origin", *shas[i:i + 40])
        for r in rs:
            play = WORK / "play" / r["instance_id"]
            shutil.rmtree(play, ignore_errors=True)
            d = play / top
            d.mkdir(parents=True)
            arch = git(g, "archive", r["base_commit"])
            subprocess.run(["tar", "-x", "-C", str(d)], input=arch, check=True)
            structure = {"structure": create_structure(str(d))}
            shutil.rmtree(play, ignore_errors=True)
            data = find_py_or_non_dict_with_path(structure["structure"],
                                                 cond=r["instance_id"].startswith("pytest-dev__"))
            patch_info = parse_patch_full(r["patch"], structure)
            changed = set()
            for fle, hunks in patch_info.items():
                for hunk in hunks:
                    if hunk["function_changed"] and hunk["newly_added"] is False:
                        if hunk["class_changed"]:
                            changed.add(f'{fle}/{hunk["class_changed"]}/{hunk["function_changed"]}')
                        else:
                            changed.add(f'{fle}/{hunk["function_changed"]}')
            if not changed:
                dropped.append(r["instance_id"])
                continue
            missing = [c for c in changed if c not in data]
            snap = f"{repo}@{r['base_commit']}"
            for uid, text in data.items():
                h = h64(text)
                txts.setdefault(h, text)
                ents.append((snap, uid, h))
            tasks.append({"instance_id": r["instance_id"], "repo": repo, "snap": snap,
                          "query": r["problem_statement"], "gold": sorted(changed), "gold_missing": missing,
                          "n_units": len(data)})
        print(f"[progress] {repo} done: tasks={len(tasks)} dropped={len(dropped)} units={len(ents)} "
              f"unique={len(txts)} {time.time() - t0:.0f}s", flush=True)
        shutil.rmtree(g, ignore_errors=True)
    snap, uid, h = zip(*ents)
    pq.write_table(pa.Table.from_pylist(tasks), out / "eval_tasks_paper.parquet", compression="zstd")
    pq.write_table(pa.table({"snap": snap, "uid": uid, "kind": ["function"] * len(uid),
                             "h": pa.array(h, pa.int64())}), out / "eval_ent_paper_000.parquet", compression="zstd")
    hs = list(txts)
    pq.write_table(pa.table({"h": pa.array(hs, pa.int64()), "path": [""] * len(hs), "text": [txts[x] for x in hs]}),
                   out / "eval_txt_paper_000.parquet", compression="zstd")
    rep = {"instances": len(tasks), "dropped_no_gold": dropped,
           "instances_with_gold_missing_in_corpus": sum(bool(t["gold_missing"]) for t in tasks),
           "units_total": len(ents), "units_unique": len(txts),
           "units_per_instance_mean": len(ents) / max(1, len(tasks)), "wall_s": time.time() - t0}
    (out / "report_paper.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "dropped_no_gold"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
