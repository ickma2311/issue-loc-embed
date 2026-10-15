"""Training-task selection. Same rules as the task-count check run before the build:
SWE-rebench V1 + V2 (Python), permissive repo licence, not an evaluation repo, non-empty issue,
patch touches a non-test .py file, V1/V2 duplicates (same repo + PR number) dropped."""
from __future__ import annotations

import re

PERMISSIVE = re.compile(
    r"^(mit|mit license|mit-0|mit-cmu license|apache license 2\.0|apache-2\.0|apache software license|"
    r"bsd.*|new bsd license|isc|isc license|the unlicense|unlicense|cc0-1\.0|zope public license 2\.1|"
    r"python software foundation license|psf-2\.0|0bsd|bsl-1\.0|boost software license 1\.0|ncsa|"
    r"university of illinois/ncsa open source license|postgresql|postgresql license|afl-3\.0|"
    r"academic free license v3\.0|historical permission notice and disclaimer|zlib|zlib license)$",
    re.I,
)
_TEST = re.compile(r"(^|/)(tests?|testing)(/|_)|_test\.py$|(^|/)test_[^/]*\.py$|conftest\.py$")


def s_(x) -> str:
    return x if isinstance(x, str) else ""


def py_nontest_files(patch: str) -> list[str]:
    files = re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)
    return [f for f in files if f.endswith(".py") and not _TEST.search(f)]


def pr_key(repo: str, iid: str):
    m = re.search(r"-(\d+)$", iid)
    return (repo.lower(), m.group(1) if m else iid)


def select(v1_rows, v2_rows, eval_repos: set[str]):
    """v1_rows / v2_rows: iterables of dicts with repo, instance_id, base_commit, license,
    problem_statement, patch (v2 already filtered to language == python).
    Returns (kept rows, stats)."""
    from collections import Counter
    stats, kept, seen = Counter(), [], set()
    for src, rows in (("v1", v1_rows), ("v2", v2_rows)):
        for r in rows:
            stats[f"{src}:all"] += 1
            repo, iid = r["repo"], r["instance_id"]
            if repo.lower() in eval_repos:
                stats[f"{src}:drop_eval_repo"] += 1; continue
            if not PERMISSIVE.match(s_(r["license"]).strip()):
                stats[f"{src}:drop_licence"] += 1; continue
            if len(s_(r["problem_statement"]).strip()) < 20:
                stats[f"{src}:drop_empty_issue"] += 1; continue
            if not py_nontest_files(s_(r["patch"])):
                stats[f"{src}:drop_no_py_src_change"] += 1; continue
            k = pr_key(repo, iid)
            if k in seen:
                stats[f"{src}:drop_dup_v1v2"] += 1; continue
            seen.add(k)
            kept.append(dict(r, src=src))
            stats[f"{src}:kept"] += 1
    return kept, stats
