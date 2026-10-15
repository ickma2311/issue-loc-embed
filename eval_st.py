"""Inference-only evaluation (Kaggle T4x2) through the user loading path (sentence-transformers).

  --set lite   SWE-Bench-Lite 274 tasks, paper protocol (the evaluated eval_*_paper files, uploaded)
  --set locb   LocBench V1 560 tasks, same protocol (build_locbench.py output, mounted kernel output)
  --model REL  the released model directory (model.safetensors sha256 checked); B0 = CodeRankEmbed (start
               point), REF = SweRankEmbed-Small, QWEN06 = Qwen3-Embedding-0.6B, from the hub

Every model: SentenceTransformer(..., trust_remote_code=True), fp16 on GPU, normalised embeddings; queries
1,024 tokens with the author's query prefix, code units 512 tokens, no prompt on code. Ranking and Acc@k
are the same as gpu_run.rank_and_save (Acc@k = every gold unit in the top k; unreachable gold = miss).
Writes $OUT_DIR/st_eval_<set>_<model>.json with per-task results, versions and input hashes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REL_SHA = "4b09febe6a337b293aa79fda038dd361d5a6ffbfd903c22a0de8e700b3f93eba"
CODE_PREFIX = "Represent this query for searching relevant code: "
QWEN_PREFIX = "Instruct: Given a GitHub issue, retrieve the code that needs to be changed to resolve it\nQuery:"
HUB = {"B0": "nomic-ai/CodeRankEmbed", "REF": "Salesforce/SweRankEmbed-Small", "QWEN06": "Qwen/Qwen3-Embedding-0.6B"}
FILES = {"lite": ("eval_tasks_paper.parquet", "eval_ent_paper_000.parquet", "eval_txt_paper_000.parquet"),
         "locb": ("locb_tasks_paper.parquet", "locb_ent_paper_000.parquet", "locb_txt_paper_000.parquet")}
N_TASKS = {"lite": 274, "locb": 560}
INPUT_SHA = {  # the evaluated Lite files (evidence bundle) and the LocBench build output (run 20261014-135654)
    "eval_tasks_paper.parquet": "9af40fae", "eval_ent_paper_000.parquet": "6b5709f3", "eval_txt_paper_000.parquet": "09f7deea",
    "locb_tasks_paper.parquet": "f69ef7f4", "locb_ent_paper_000.parquet": "317f15ff", "locb_txt_paper_000.parquet": "27eedda7"}
# LocBench tasks from repositories with a training-set counterpart, excluded from the headline numbers:
# sktime/sktime (fork aeon-toolkit/aeon in training), Qiskit/qiskit (= Qiskit/qiskit-terra in training, renamed)
LOCB_EXCL = {"sktime": {"sktime/sktime"}, "sktime_qiskit": {"sktime/sktime", "qiskit/qiskit"}}
Q_LEN, D_LEN = 1024, 512


def log(m):
    print(f"[eval] {time.strftime('%H:%M:%S')} {m}", flush=True)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_one(root: str, name: str) -> Path:
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    assert len(hits) == 1, f"{name}: expected one file under {root}, found {hits}"
    return Path(hits[0])


def load_set(which: str, root: str, limit: int, check_inputs: bool = True):
    paths = [find_one(root, f) for f in FILES[which]]
    tasks = pq.read_table(paths[0]).to_pylist()
    tasks.sort(key=lambda t: t["instance_id"])
    if not limit:
        assert len(tasks) == N_TASKS[which], (which, len(tasks))
    else:
        tasks = tasks[:limit]
    snaps = {t["snap"] for t in tasks}
    ent = pq.read_table(paths[1], columns=["snap", "uid", "h"]).to_pydict()
    by_snap = defaultdict(list)
    for s, u, h in zip(ent["snap"], ent["uid"], ent["h"]):
        if s in snaps:
            by_snap[s].append((u, h))
    del ent
    need = sorted({h for v in by_snap.values() for _, h in v})
    want = set(need)
    text = {}
    for b in pq.ParquetFile(paths[2]).iter_batches(batch_size=200000, columns=["h", "text"]):
        for h, x in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            if h in want:
                text[h] = x
    assert len(text) == len(need), (len(text), len(need))
    inputs = {p.name: sha256(p) for p in paths}
    for k, v in inputs.items() if check_inputs else ():
        assert v.startswith(INPUT_SHA[k]), f"{k}: sha256 {v[:8]} != expected {INPUT_SHA[k]}"
    return tasks, by_snap, need, text, inputs


def load_st(model: str, rel_dir: str, device: str, dtype: str):
    import torch
    from sentence_transformers import SentenceTransformer
    info = {}
    if model == "REL":
        d = Path(rel_dir)
        info["path"] = str(d)
        info["safetensors_sha256"] = sha256(d / "model.safetensors")
        assert info["safetensors_sha256"] == REL_SHA, info
        target, kw = str(d), {}
    else:
        from huggingface_hub import snapshot_download
        from issueloc.revisions import model_rev
        target = HUB[model]
        info["revision"] = model_rev(target)
        snap = snapshot_download(target, revision=info["revision"])
        kw = {"revision": info["revision"]}
    st = SentenceTransformer(target, device=device, trust_remote_code=True,
                             model_kwargs={"torch_dtype": getattr(torch, dtype)}, **kw)
    cast = getattr(torch, dtype)
    st.to(cast)  # the nomic-bert remote code ignores torch_dtype; cast explicitly and check
    st.eval()
    pd = {p.dtype for p in st.parameters()}
    assert pd == {cast}, pd
    info["param_dtype"] = str(pd.pop())
    # the weights in use are exactly the files on disk: the release file, or the hub snapshot at the revision
    from safetensors.torch import load_file
    wdir = target if model == "REL" else snap
    files = sorted(glob.glob(os.path.join(wdir, "*.safetensors")))
    assert files, f"no safetensors in {wdir}"
    ref = {}
    for f in files:
        ref.update(load_file(f))
    msd = st[0].auto_model.state_dict()
    norm = lambda k: k[len("model."):] if k.startswith("model.") and k[len("model."):] in msd else k  # noqa: E731
    ref = {norm(k): v for k, v in ref.items()}
    absent = [k for k in msd if k not in ref]
    unused = [k for k in ref if k not in msd]
    assert not absent and not unused, (absent[:5], unused[:5])
    bad = [k for k in ref if not torch.equal(msd[k].float().cpu(), ref[k].to(cast).float())]
    assert not bad, bad[:5]
    info["weights_check"] = f"{len(ref)} tensors equal to {'model.safetensors' if model == 'REL' else 'hub snapshot'} ({dtype})"
    return st, info


def encode(models, texts: list[str], max_len: int, prompt: str | None, bs: int, tag: str) -> np.ndarray:
    """Length-sorted, split across the per-GPU models (one thread each)."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = [order[k::len(models)] for k in range(len(models))]
    out, errs = [None] * len(texts), [None] * len(models)
    t0 = time.time()
    done = [0] * len(models)

    def work(k):
        try:
            m = models[k]
            m.max_seq_length = max_len
            idx = parts[k]
            for s in range(0, len(idx), 20000):
                chunk = idx[s:s + 20000]
                E = m.encode([texts[i] for i in chunk], batch_size=bs, prompt=prompt, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
                if not np.isfinite(E).all():
                    raise FloatingPointError(f"{tag}: non-finite embeddings in chunk at {s}")
                for i, e in zip(chunk, E):
                    out[i] = e.astype(np.float16)
                done[k] += len(chunk)
                el = time.time() - t0
                tot = sum(done)
                print(f"[progress] {tag} {tot}/{len(texts)} {el:.0f}s eta {el / tot * (len(texts) - tot):.0f}s", flush=True)
        except BaseException as e:  # re-raised below
            errs[k] = e

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(models))]
    [t.start() for t in th]
    [t.join() for t in th]
    for e in errs:
        if e is not None:
            raise e
    E = np.stack(out)
    assert np.isfinite(E).all(), f"{tag}: non-finite embeddings"
    return E


def rank(tasks, by_snap, D, Q, need):
    row = {h: i for i, h in enumerate(need)}
    per = []
    for i, t in enumerate(tasks):
        units = by_snap[t["snap"]]
        sims = D[[row[h] for _, h in units]].astype(np.float32) @ Q[i].astype(np.float32)
        best = {}
        for (u, _), s in zip(units, sims):
            if s > best.get(u, -9):
                best[u] = s
        ranked = sorted(best, key=best.get, reverse=True)
        pos = {u: j + 1 for j, u in enumerate(ranked)}
        gold = list(t["gold"])
        fbest = {}
        for u, s in best.items():
            f = u.split(":", 1)[0]
            fbest[f] = max(fbest.get(f, -9), s)
        fpos = {f: j + 1 for j, f in enumerate(sorted(fbest, key=fbest.get, reverse=True))}
        per.append({"instance_id": t["instance_id"], "repo": t["repo"],
                    "func_need_k": int(max(pos.get(g, 10**9) for g in gold)),
                    "file_need_k": int(max(fpos.get(g.split(":", 1)[0], 10**9) for g in gold)),
                    "gold_missing": int(sum(g not in pos for g in gold)), "n_units": len(ranked)})
    return per


def summary(per):
    acc = lambda key, k: 100 * float(np.mean([p[key] <= k for p in per])) if per else None  # noqa: E731
    return {"n": len(per), "func_acc": {k: acc("func_need_k", k) for k in (1, 5, 10, 15, 20)},
            "file_acc": {k: acc("file_need_k", k) for k in (1, 5, 10)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["lite", "locb"], required=True)
    ap.add_argument("--model", choices=["REL", "B0", "REF", "QWEN06"], required=True)
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--rel-dir", default="/kaggle/temp/issue-loc-embed-small")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--no-input-check", action="store_true", help="rebuilt corpus: skip the SHA-256 check of our files")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-units", type=int, default=0, help="smoke only: cap units per snapshot")
    a = ap.parse_args()
    import torch
    import sentence_transformers
    import transformers
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks, by_snap, need, text, inputs = load_set(a.set, a.root, a.limit, not a.no_input_check)
    if a.max_units:
        by_snap = {s: v[: a.max_units] for s, v in by_snap.items()}
        need = sorted({h for v in by_snap.values() for _, h in v})
    log(f"{a.set} {a.model}: {len(tasks)} tasks, {len(need)} unique units")
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    dtype = a.dtype if devs[0] != "cpu" else "float32"
    loaded = [load_st(a.model, a.rel_dir, d, dtype) for d in devs]
    models, info = [m for m, _ in loaded], loaded[0][1]
    prompt = QWEN_PREFIX if a.model == "QWEN06" else CODE_PREFIX
    bs = 32 if a.model == "QWEN06" else 128
    t0 = time.time()
    D = encode(models, [text[h] for h in need], D_LEN, None, bs, f"{a.set}-{a.model}-units")
    Q = encode(models, [t["query"] for t in tasks], Q_LEN, prompt, 16, f"{a.set}-{a.model}-queries")
    enc_s = time.time() - t0
    per = rank(tasks, by_snap, D, Q, need)
    res = {"set": a.set, "model": a.model, **info, "n_tasks": len(tasks), "n_units": len(need),
           "inputs_sha256": inputs, "dtype": dtype, "max_len": {"query": Q_LEN, "unit": D_LEN},
           "query_prompt": prompt, "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                                "sentence_transformers": sentence_transformers.__version__,
                                                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
           "encode_seconds": enc_s, "smoke": bool(a.limit or a.max_units),
           "all": summary(per), "per_instance": per}
    if a.set == "locb":
        for k, repos in LOCB_EXCL.items():
            res[f"excl_{k}"] = summary([p for p in per if p["repo"].lower() not in repos])
    f = out / f"st_eval_{a.set}_{a.model}.json"
    f.write_text(json.dumps(res, indent=1))
    log(f"{a.set} {a.model}: func Acc@10 {res['all']['func_acc'][10]:.2f} "
        f"(headline {res.get('excl_sktime_qiskit', {}).get('func_acc', {}).get(10)}) encode {enc_s:.0f}s -> {f.name}")


if __name__ == "__main__":
    sys.exit(main())
