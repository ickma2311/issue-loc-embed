"""Kaggle CPU: build the SWE-bench Multilingual evaluation set from mteb/SWEbenchMultilingualRR at a pinned revision.
The candidates and gold labels are used as released (one candidate list per query; text = "path/Class/function" + code).
Identical candidate texts are stored once (h = blake2b-64 of the text, as issueloc.paperfmt.h64).
Language and repository per query come from data/ml_meta.json (scripts/multilang_meta.py).
Writes $OUT_DIR/ml_tasks.parquet (instance_id, repo, language, snap, query, gold), ml_ent.parquet (snap, uid, h),
ml_txt.parquet (h, text), ml_build.json (counts, sha256 of every downloaded file).
Usage: python build_multilang.py [--meta ml_meta.json]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import re
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO, REV = "mteb/SWEbenchMultilingualRR", "2129d0fd2fe31e95b4eecbeca873b623d7f58c4d"
FILES = ["queries/train-00000-of-00001.parquet", "qrels/train-00000-of-00001.parquet"] + \
        [f"corpus/train-{i:05d}-of-00012.parquet" for i in range(12)]
# language of a candidate = extension in its "path/Class/function" header; groups merge languages with tiny counts
EXT = {"c": "C", "h": "C", "cc": "C++", "cpp": "C++", "cxx": "C++", "hpp": "C++", "hh": "C++", "go": "Go",
       "java": "Java", "js": "JavaScript", "jsx": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript",
       "ts": "TypeScript", "tsx": "TypeScript", "php": "PHP", "rb": "Ruby", "rs": "Rust"}
GROUP = {"C": "C/C++", "C++": "C/C++", "JavaScript": "JS/TS", "TypeScript": "JS/TS"}


def header_lang(text: str):
    m = re.search(r"\.([A-Za-z0-9]+)/", text.split("\n", 1)[0])
    return EXT.get(m.group(1).lower()) if m else None


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little", signed=True)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(dst: Path) -> dict:
    shas = {}
    for f in FILES:
        p = dst / f
        p.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{f}", p)
                break
            except OSError as e:
                if attempt == 3:
                    raise
                print(f"[progress] download {f} failed ({e}); retry", flush=True); time.sleep(10 * (attempt + 1))
        shas[f] = sha256(p)
        print(f"[progress] download {f} {p.stat().st_size / 1e6:.0f} MB {time.time() - t0:.0f}s", flush=True)
    return shas


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--meta", default="data/ml_meta.json")
    a = ap.parse_args()
    out = Path(os.environ.get("OUT_DIR", "out")); out.mkdir(parents=True, exist_ok=True)
    raw = Path("/kaggle/temp/mteb_ml") if Path("/kaggle").exists() else out / "raw"
    shas = fetch(raw)
    meta = json.loads(Path(a.meta).read_text())["queries"]
    q = pq.read_table(raw / FILES[0]).to_pylist()
    assert len(q) == len(meta) == 234, (len(q), len(meta))
    qr = pq.read_table(raw / FILES[1]).to_pydict()
    cand = defaultdict(set); gold = defaultdict(list)
    for qid, cid, sc in zip(qr["query-id"], qr["corpus-id"], qr["score"]):
        cand[qid].add(cid)
        if sc > 0:
            gold[qid].append(cid)
    owner = {c: qid for qid, cs in cand.items() for c in cs}
    assert len(owner) == sum(len(v) for v in cand.values()), "a corpus id belongs to two queries"
    ents, txt, seen_ids = [], {}, set()
    n_rows, gold_set, gold_lang = 0, {c for v in gold.values() for c in v}, collections.defaultdict(collections.Counter)
    for f in FILES[2:]:
        for b in pq.ParquetFile(raw / f).iter_batches(batch_size=50000, columns=["id", "title", "text"]):
            assert all(not x for x in b.column(1).to_pylist()), f"{f}: non-empty title (MTEB joins title and text)"
            for cid, t in zip(b.column(0).to_pylist(), b.column(2).to_pylist()):
                n_rows += 1
                if cid not in owner:
                    continue
                if cid in gold_set:
                    gold_lang[owner[cid]][header_lang(t)] += 1
                hv = h64(t)
                txt.setdefault(hv, t)
                ents.append((owner[cid], cid, hv)); seen_ids.add(cid)
        print(f"[progress] corpus {f}: rows {n_rows}, kept {len(ents)}, unique texts {len(txt)}", flush=True)
    missing = sum(1 for c in owner if c not in seen_ids)
    assert missing == 0, f"{missing} qrels corpus ids have no text"
    lang = {}
    for qid, c in gold_lang.items():
        c.pop(None, None)
        lang[qid] = c.most_common(1)[0][0] if c else meta[qid]["language"]
    tasks = [{"instance_id": x["id"], "repo": meta[x["id"]]["repo"], "language": lang[x["id"]],
              "language_group": GROUP.get(lang[x["id"]], lang[x["id"]]), "language_from_patch": meta[x["id"]]["language"],
              "swebench_instance": meta[x["id"]]["instance_id"], "snap": x["id"], "query": x["text"],
              "gold": sorted(gold[x["id"]])} for x in q]
    assert all(t["gold"] for t in tasks), "a query has no gold"
    pq.write_table(pa.Table.from_pylist(tasks), out / "ml_tasks.parquet")
    s, u, h = zip(*ents)
    pq.write_table(pa.table({"snap": list(s), "uid": list(u), "h": pa.array(h, pa.int64())}), out / "ml_ent.parquet")
    hs = list(txt)
    pq.write_table(pa.table({"h": pa.array(hs, pa.int64()), "text": pa.array([txt[x] for x in hs], pa.large_string())}),
                   out / "ml_txt.parquet")
    res = {"repo": REPO, "revision": REV, "downloaded_sha256": shas, "n_queries": len(tasks), "corpus_rows": n_rows,
           "candidates": len(ents), "unique_texts": len(txt), "gold_total": sum(len(t["gold"]) for t in tasks),
           "languages": dict(collections.Counter(t["language"] for t in tasks).most_common()),
           "language_groups": dict(collections.Counter(t["language_group"] for t in tasks).most_common()),
           "language_changed_vs_patch": [(t["instance_id"], t["language_from_patch"], t["language"]) for t in tasks
                                         if t["language"] != t["language_from_patch"]],
           "outputs_sha256": {p: sha256(out / p) for p in ("ml_tasks.parquet", "ml_ent.parquet", "ml_txt.parquet")}}
    (out / "ml_build.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "downloaded_sha256"}), flush=True)


if __name__ == "__main__":
    main()
