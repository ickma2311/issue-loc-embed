"""Convert build_data.py output to the SweRank / CoRNStack paper protocol:
units = functions + methods of non-test files (CoRNStack rule), text = "<path>/<Class>/<func>\\n" + code
(methods: "class C:" + code, as cornstack find_py_or_non_dict_with_path), LocAgent-convention class labels
mapped to Class.__init__. Writes *_paper files next to the build_data.py layout so gpu_run (EVAL_SET=paper) reads them.
Usage: python -m issueloc.paperfmt <c1_data_dir> <out_dir>
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def is_test_cornstack(path: str, repo: str) -> bool:
    fname = path.rsplit("/", 1)[-1].lower()
    if "test" in fname:
        return True
    return "test" in path.lower() and repo != "pytest-dev/pytest"


def header_text(uid: str, kind: str, code: str) -> str:
    path, qual = uid.split(":", 1)
    parts = qual.split(".")
    fp = f"{path}/{'/'.join(parts)}"
    if kind == "method":
        return f"{fp}\n" + f"class {parts[-2]}:" + code.rstrip("\n")
    return f"{fp}\n" + code.rstrip("\n")


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little", signed=True)


def map_label(uid: str, kind_of: dict) -> str | None:
    """LocAgent label -> paper-corpus unit id (class -> Class.__init__)."""
    k = kind_of.get(uid)
    if k == "class":
        init = uid + ".__init__"
        return init if kind_of.get(init) == "method" else None
    return uid if k in ("function", "method") else None


def _convert(tasks, ent_files, txt_files, out: Path, prefix: str, label_keys, drop_repos=(), eval_mode=False,
             variant: str = "V1"):
    """variant (eval only, robustness checks): V1 paper protocol; V2 = same units, code only;
    V3 = V1 text, test files included."""
    snaps = {t["snap"]: t["repo"] for t in tasks if t["repo"] not in drop_repos}
    code = {}
    for f in txt_files:
        t = pq.read_table(f).to_pydict()
        code.update(zip(t["h"], t["text"]))
    kinds = defaultdict(dict)       # snap -> {uid: kind}
    rows, texts = [], {}
    for f in ent_files:
        e = pq.read_table(f).to_pydict()
        for s, u, k, h in zip(e["snap"], e["uid"], e["kind"], e["h"]):
            if s not in snaps:
                continue
            kinds[s][u] = k
            if k == "class" or (variant != "V3" and is_test_cornstack(u.split(":", 1)[0], snaps[s])):
                continue
            txt = code[h] if variant == "V2" else header_text(u, k, code[h])  # V2 = raw code, as in calibration
            nh = h64(txt)
            texts.setdefault(nh, txt)
            rows.append((s, u, k, nh))
    del code
    pool_n = defaultdict(int)
    for s, *_ in rows:
        pool_n[s] += 1
    in_pool = defaultdict(set)
    for s, u, _, _ in rows:
        in_pool[s].add(u)
    out_tasks, stats = [], defaultdict(int)
    for t in tasks:
        if t["snap"] not in snaps:
            stats["dropped_repo"] += 1
            continue
        t = dict(t)
        if not eval_mode:  # exclusion set keeps the original LocAgent labels so class <-> method relations still apply
            t["excl_positives"] = sorted(set(t.get("all_positives") or []))
            stats["pos_from_class_label"] += sum(kinds[t["snap"]].get(x) == "class" for x in t["positives"])
        for key in label_keys:
            if eval_mode:  # same as calibration "gold_init": class -> Class.__init__ even if absent (= unreachable)
                mapped = sorted({x + ".__init__" if kinds[t["snap"]].get(x) == "class" else x for x in t[key]})
            else:
                mapped = [map_label(x, kinds[t["snap"]]) for x in t[key]]
                mapped = sorted({x for x in mapped if x and x in in_pool[t["snap"]]})
            stats[f"{key}_lost"] += len(t[key]) - len(mapped)
            t[key] = mapped
        if "full_est" in t:
            old = max(1, t["pool_n"])
            t["full_est"] = max(pool_n[t["snap"]], int(round(t["full_est"] * pool_n[t["snap"]] / old)))
            t["pool_n"] = pool_n[t["snap"]]
        if not eval_mode:
            t["excl_positives"] = sorted(set(t["excl_positives"]) | set(t["all_positives"]))
        if not t[label_keys[0]]:
            stats["dropped_no_label"] += 1
            continue
        out_tasks.append(t)
    s, u, k, h = zip(*rows)
    sfx = "_paper" if variant == "V1" else f"_paper{variant}"
    pq.write_table(pa.Table.from_pylist(out_tasks), out / f"{prefix}_tasks{sfx}.parquet", compression="zstd")
    pq.write_table(pa.table({"snap": s, "uid": u, "kind": k, "h": pa.array(h, pa.int64())}),
                   out / f"{prefix}_ent{sfx}_000.parquet", compression="zstd")
    hs = list(texts)
    pq.write_table(pa.table({"h": pa.array(hs, pa.int64()), "path": [""] * len(hs), "text": [texts[x] for x in hs]}),
                   out / f"{prefix}_txt{sfx}_000.parquet", compression="zstd")
    stats.update(tasks=len(out_tasks), units=len(rows), unique_texts=len(texts))
    return dict(stats)


def main(src: str, dst: str):
    src, out = Path(src), Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((src / "report.json").read_text())
    drop = set(rep.get("fork_like_train_repos", []))
    assert len(drop) == 13, f"expected 13 eval-name-sharing repos to drop, got {len(drop)}"
    (out / "report.json").write_text(json.dumps({"fork_like_train_repos": sorted(drop)}))
    ev = pq.read_table(src / "eval_tasks.parquet").to_pylist()
    for t in ev:
        t["gold"] = list(t["gold"])
    st_e = _convert(ev, [src / "eval_ent_000.parquet"], [src / "eval_txt_000.parquet"], out, "eval", ["gold"], eval_mode=True)
    assert st_e["tasks"] == 274 and st_e["unique_texts"] == 138308, st_e  # = calibration V1 + gold_init
    st_v = {}
    for v in ("V2", "V3"):  # robustness variants
        st_v[v] = _convert(ev, [src / "eval_ent_000.parquet"], [src / "eval_txt_000.parquet"], out, "eval", ["gold"],
                           eval_mode=True, variant=v)
    assert st_v["V2"]["unique_texts"] == 130609 and st_v["V3"]["unique_texts"] == 257150, st_v  # = calibration
    tr = pq.read_table(src / "train_tasks.parquet").to_pylist()
    for t in tr:
        t["positives"], t["all_positives"] = list(t["positives"]), list(t.get("all_positives") or t["positives"])
    st_t = _convert(tr, sorted(glob.glob(str(src / "train_ent_*.parquet"))), sorted(glob.glob(str(src / "train_txt_*.parquet"))),
                    out, "train", ["positives", "all_positives"], drop_repos=drop)
    (out / "report_paperfmt.json").write_text(json.dumps({"eval": st_e, "eval_variants": st_v, "train": st_t,
                                                          "dropped_repos": sorted(drop)}, indent=1))
    print(json.dumps({"eval": st_e, "train": st_t}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
