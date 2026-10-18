"""Merge the Python and the multi-language training data into one data set (Kaggle CPU).
1. issueloc.paperfmt on the v0.1 Python build (build_data.py output), exactly as v0.1 (asserts inside).
2. Add the multi-language build (build_data_ml.py output: train_tasks_ml / train_ent_ml_* / train_txt_ml_*).
Writes $OUT_DIR/data: train_tasks_paper.parquet (Python + other languages, column "language"),
train_ent_paper_000.parquet (Python) and _001... (other languages), ONE train_txt_paper_000.parquet (gpu_run.mine_lowmem
reads one text file), the Lite eval files of paperfmt, merge_report.json with counts and sha256.
Usage: INPUT_ROOT=<folder with build/data and build_ml/data> OUT_DIR=merged python merge_data.py
"""
import glob, hashlib, json, os, sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from issueloc import paperfmt


def one(name):
    root = os.environ.get("INPUT_ROOT", "/kaggle/input")  # folder with the two build outputs
    hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
    assert len(hits) == 1, (name, hits)
    return Path(hits[0])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(os.environ.get("OUT_DIR", "out")) / "data"
    out.mkdir(parents=True, exist_ok=True)
    c1 = one("train_tasks.parquet").parent
    paperfmt.main(str(c1), str(out))
    py_tasks = pq.read_table(out / "train_tasks_paper.parquet").to_pylist()
    assert len(py_tasks) == 18784, len(py_tasks)  # v0.1 training set (18,784 tasks)
    for t in py_tasks:
        t["language"] = "python"
    ml_dir = one("train_tasks_ml.parquet").parent
    ml_tasks = pq.read_table(ml_dir / "train_tasks_ml.parquet").to_pylist()
    py_snaps, ml_snaps = {t["snap"] for t in py_tasks}, {t["snap"] for t in ml_tasks}
    assert not (py_snaps & ml_snaps)
    keys = ["instance_id", "repo", "language", "snap", "query", "positives", "all_positives", "excl_positives",
            "n_pos_raw", "pool_n", "full_est"]
    rows = [{k: t.get(k) for k in keys} for t in py_tasks + ml_tasks]
    ids = [r["instance_id"] for r in rows]
    assert len(ids) == len(set(ids))
    pq.write_table(pa.Table.from_pylist(rows), out / "train_tasks_paper.parquet", compression="zstd")
    for i, f in enumerate(sorted(ml_dir.glob("train_ent_ml_*.parquet")), start=1):
        t = pq.read_table(f, columns=["snap", "uid", "kind", "h"])
        pq.write_table(t, out / f"train_ent_paper_{i:03d}.parquet", compression="zstd")
    # one text file: Python texts first, then the other languages (dedup by h)
    py_txt = pq.read_table(out / "train_txt_paper_000.parquet", columns=["h", "path", "text"])
    seen = set(py_txt.column("h").to_pylist())
    parts = [py_txt.cast(pa.schema([("h", pa.int64()), ("path", pa.string()), ("text", pa.large_string())]))]
    n_ml = 0
    for f in sorted(ml_dir.glob("train_txt_ml_*.parquet")):
        t = pq.read_table(f, columns=["h", "path", "text"]).to_pydict()
        keep = [i for i, h in enumerate(t["h"]) if h not in seen]
        seen.update(t["h"][i] for i in keep)
        n_ml += len(keep)
        parts.append(pa.table({"h": pa.array([t["h"][i] for i in keep], pa.int64()), "path": pa.array([""] * len(keep), pa.string()),
                               "text": pa.array([t["text"][i] for i in keep], pa.large_string())}))
    allt = pa.concat_tables(parts)
    pq.write_table(allt, out / "train_txt_paper_000.parquet", compression="zstd", row_group_size=100000)
    # every pool unit has a text
    need = set()
    for f in sorted(out.glob("train_ent_paper_*.parquet")):
        need |= set(pq.read_table(f, columns=["h"]).column("h").to_pylist())
    assert need <= seen, len(need - seen)
    rep = {"python_tasks": len(py_tasks), "ml_tasks": len(ml_tasks), "tasks": len(rows),
           "per_language": {L: sum(r["language"] == L for r in rows) for L in sorted({r["language"] for r in rows})},
           "texts": allt.num_rows, "ml_new_texts": n_ml, "pool_units_with_text": len(need),
           "sha256": {f.name: sha(f) for f in sorted(out.glob("*.parquet"))}}
    (out / "merge_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
