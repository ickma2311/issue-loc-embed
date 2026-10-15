"""Export the training data as pointers only (no code, no issue text): one row per training task with the
SWE-rebench instance id, repository, base commit, the positive unit ids, and whether each positive passed
the consistency filter. Texts are rebuilt with build_data.py (+ issueloc.paperfmt) from the public sources.
Usage: python scripts/export_pointers.py <data_dir_with_train_tasks_paper.parquet> <out.parquet> [kept_it1.json]
Without kept_it1.json the filter flag is omitted (re-running mining reproduces it: 43,068 kept pairs)."""
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main(data_dir, out, kept_json=None):
    kept = set(json.load(open(kept_json))["kept"]) if kept_json else None
    rows = []
    for t in pq.read_table(f"{data_dir}/train_tasks_paper.parquet").to_pylist():
        repo, sha = t["snap"].split("@", 1)
        rows.append({
            "instance_id": t["instance_id"],          # SWE-rebench / SWE-rebench-V2 instance id
            "repo": repo,
            "base_commit": sha,
            "positives": list(t["positives"]),        # "path.py:Class.method" unit ids at base_commit
        })
        if kept is not None:
            rows[-1]["positives_kept"] = [f"{t['instance_id']}|{p}" in kept for p in t["positives"]]
    pq.write_table(pa.Table.from_pylist(rows), out, compression="zstd")
    print(f"{len(rows)} tasks, {len({r['repo'] for r in rows})} repositories, "
          f"{sum(len(r['positives']) for r in rows)} positives -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
