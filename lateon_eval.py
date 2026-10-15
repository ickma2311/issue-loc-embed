"""Evaluate the permissive late-interaction competitors (LightOn LateOn-Code / -edge, Apache-2.0)
on the same SWE-Bench-Lite corpus, gold and metric code as our models (gpu_run.rank_and_save).

PyLate defaults of each model are kept ([Q]/[D] prefixes, query length 256, skiplist); documents are
truncated at 512 tokens like every other model in our evaluation. Scoring is MaxSim. Units are encoded
repository by repository so the token vectors of the whole corpus never sit in memory at once.
Usage: python lateon_eval.py --model LATEON|LATEON_EDGE --rep R1|R2 [--limit N] [--smoke-units N]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gpu_run import eval_units, find_data, log, rank_and_save, rep_text
from issueloc.revisions import model_rev

MODELS = {"LATEON": "lightonai/LateOn-Code", "LATEON_EDGE": "lightonai/LateOn-Code-edge"}


def load(name: str, device: str):
    from pylate import models
    return models.ColBERT(model_name_or_path=name, revision=model_rev(name), document_length=512, device=device)


def _np16(x):
    x = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise FloatingPointError("non-finite token embeddings")
    return x.astype(np.float16)


def encode(models, texts: list[str], is_query: bool, progress: str) -> list[np.ndarray]:
    """Token-vector matrices in input order; one model copy (and thread) per GPU."""
    n = len(texts)
    out: list = [None] * n
    order = sorted(range(n), key=lambda i: len(texts[i]))
    parts = [order[k::len(models)] for k in range(len(models))]
    errors = [None] * len(models)
    t0 = time.time()

    def work(k):
        try:
            idx = parts[k]
            for s in range(0, len(idx), 2048):
                chunk = idx[s:s + 2048]
                embs = models[k].encode([texts[i] for i in chunk], is_query=is_query, batch_size=64,
                                        show_progress_bar=False)
                for i, e in zip(chunk, embs):
                    out[i] = _np16(e)
                if k == 0:
                    el = time.time() - t0
                    done = min(len(idx), s + 2048) * len(models)
                    print(f"[progress] {progress} {done}/{n} {el:.0f}s eta {el / done * max(0, n - done):.0f}s",
                          flush=True)
        except BaseException as e:
            errors[k] = e

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(models))]
    [t.start() for t in th]
    [t.join() for t in th]
    for e in errors:
        if e is not None:
            raise e
    assert all(o is not None for o in out)
    return out


@torch.no_grad()
def maxsim(q: np.ndarray, docs: list[np.ndarray], device: str, chunk: int = 1024) -> np.ndarray:
    qt = torch.from_numpy(q).to(device)
    scores = np.zeros(len(docs), dtype=np.float32)
    order = sorted(range(len(docs)), key=lambda i: docs[i].shape[0])
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        L = max(1, max(docs[i].shape[0] for i in idx))
        D = torch.zeros(len(idx), L, q.shape[1], dtype=torch.float16)
        mask = torch.zeros(len(idx), L, dtype=torch.bool)
        for r, i in enumerate(idx):
            d = docs[i]
            D[r, :d.shape[0]] = torch.from_numpy(d)
            mask[r, :d.shape[0]] = True
        D, mask = D.to(device), mask.to(device)
        sim = torch.einsum("qd,bld->bql", qt.float(), D.float())
        sim = sim.masked_fill(~mask[:, None, :], float("-inf")).amax(-1)
        scores[idx] = sim.sum(-1).cpu().numpy()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--rep", choices=["R1", "R2"], required=True)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke-units", type=int, default=0, help="CPU smoke only: cap units per snapshot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = find_data(a.data_root)
    tasks, by_snap, txt = eval_units(data, a.limit)
    if a.smoke_units:
        for s in by_snap:
            gold = {g for t in tasks if t["snap"] == s for g in t["gold"]}
            keep = [x for x in by_snap[s] if x[0] in gold] + [x for x in by_snap[s] if x[0] not in gold]
            by_snap[s] = keep[:a.smoke_units]
    hpos = {h: i for i, h in enumerate(txt["h"])}
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    models = [load(MODELS[a.model], d) for d in devs]
    name = f"{a.model}_{a.rep}"
    log(f"{name}: {len(tasks)} tasks on {devs}")
    Q = encode(models, [t["query"] for t in tasks], True, f"{name} queries")
    # group tasks by repository; encode that repository's unique units once
    by_repo = defaultdict(list)
    for i, t in enumerate(tasks):
        by_repo[t["repo"]].append(i)
    scores_by_task = {}
    for repo, ids in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        hs = sorted({h for i in ids for _, h in by_snap[tasks[i]["snap"]]})
        texts = [rep_text(a.rep, txt["path"][hpos[h]], txt["text"][hpos[h]]) for h in hs]
        log(f"{name}: {repo} {len(ids)} tasks, {len(hs)} unique units")
        E = encode(models, texts, False, f"{name} {repo}")
        row = {h: j for j, h in enumerate(hs)}
        for i in ids:
            units = by_snap[tasks[i]["snap"]]
            scores_by_task[i] = maxsim(Q[i], [E[row[h]] for _, h in units], devs[0])
        del E, texts
    rank_and_save(tasks, by_snap, lambda i, units: scores_by_task[i], data, out, name, MODELS[a.model], a.rep)


if __name__ == "__main__":
    sys.exit(main())
