"""Metadata for the multi-language benchmark (SWE-bench Multilingual): map each mteb/SWEbenchMultilingualRR query to its
SWE-bench_Multilingual instance (repo, language), and check the repositories against our training data.
Language = the most common source-file language among the files changed by the fix (test files excluded).
Usage: python -I scripts/multilang_meta.py <queries.parquet> <SWE-bench_Multilingual test parquet> <train_pointers.parquet> > data/ml_meta.json
"""
import collections
import json
import re
import sys

import pyarrow.parquet as pq

EXT = {".c": "C", ".h": "C/C++ header", ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
       ".go": "Go", ".java": "Java", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
       ".ts": "TypeScript", ".tsx": "TypeScript", ".php": "PHP", ".rb": "Ruby", ".rs": "Rust"}


def norm(x):
    return re.sub(r"\s+", " ", x).strip()[:300]


def main(qf, sf, tf):
    q = pq.read_table(qf).to_pylist()
    s = pq.read_table(sf).to_pylist()
    by = collections.defaultdict(list)
    for r in s:
        by[norm(r["problem_statement"])].append(r)
    train = {r.lower() for r in pq.read_table(tf, columns=["repo"]).column(0).to_pylist()}
    train_names = {r.split("/")[1] for r in train}
    out, langs = {}, collections.Counter()
    for x in q:
        m = by[norm(x["text"])]
        assert len(m) == 1, (x["id"], len(m))
        r = m[0]
        files = re.findall(r"^diff --git a/(\S+)", r["patch"], re.M)
        c = collections.Counter()
        for f in files:
            if re.search(r"(^|/)(test|tests|spec|__tests__)/|_test\.|\.test\.|_spec\.", f):
                continue
            ext = "." + f.rsplit(".", 1)[-1] if "." in f.rsplit("/", 1)[-1] else ""
            if ext in EXT:
                c[EXT[ext]] += 1
        lang = c.most_common(1)[0][0] if c else "unknown"
        if lang == "C/C++ header":
            lang = "C" if "C" in c else ("C++" if "C++" in c else lang)
        out[x["id"]] = {"instance_id": r["instance_id"], "repo": r["repo"], "language": lang, "base_commit": r["base_commit"],
                        "created_at": r["created_at"]}
        langs[lang] += 1
    repos = sorted({v["repo"] for v in out.values()})
    leak = [r for r in repos if r.lower() in train or r.split("/")[1].lower() in train_names]
    print(json.dumps({"n": len(out), "languages": dict(langs.most_common()), "repos": repos, "n_repos": len(repos),
                      "training_overlap_by_name": leak, "queries": out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
