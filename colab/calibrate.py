"""Colab (L4): which corpus protocol reproduces the paper's numbers for B0 (CodeRankEmbed, 58.76) and
REF (SweRankEmbed-Small, 74.45)? Inference only. Corpora are derived from the build_data.py eval files (LocAgent
gold, LocAgent-style unit ids):
  V1  functions + methods, non-test files (CoRNStack rule), text "<path>/<Class>/<func>\\n" (+ "class C:")
      + code, as in CoRNStack find_py_or_non_dict_with_path      <- best guess of the paper protocol
  V2  same units, code only
  V3  functions + methods of all files (tests included), V1 text
Gold: LocAgent edit_functions. Class-level gold items are scored two ways: "drop" (removed; instances
left without gold are excluded) and "fail" (kept, can never be retrieved).
Usage: python colab/calibrate.py --data DIR --models B0,REF [--variants V1,V2,V3] --out DIR
       python colab/calibrate.py ... --models LATEON,LATEON_EDGE --variants V1   (PyLate, MaxSim)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (issueloc)
from issueloc.revisions import model_rev  # noqa: E402

QUERY_PREFIX = "Represent this query for searching relevant code: "
HF = {"B0": "nomic-ai/CodeRankEmbed", "REF": "Salesforce/SweRankEmbed-Small",
      "LATEON": "lightonai/LateOn-Code", "LATEON_EDGE": "lightonai/LateOn-Code-edge"}
PAPER = {"B0": 58.76, "REF": 74.45}


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


def build(data: Path):
    tasks = pq.read_table(data / "eval_tasks.parquet").to_pylist()
    assert len(tasks) == 274
    ent = pq.read_table(data / "eval_ent_000.parquet").to_pydict()
    txt = pq.read_table(data / "eval_txt_000.parquet").to_pydict()
    code = dict(zip(txt["h"], txt["text"]))
    snaps = {t["snap"]: t["repo"] for t in tasks}
    kinds = {}
    variants = {v: defaultdict(list) for v in ("V1", "V2", "V3")}
    for s, u, k, h in zip(ent["snap"], ent["uid"], ent["kind"], ent["h"]):
        if s not in snaps:
            continue
        kinds[(s, u)] = k
        if k == "class":
            continue
        test = is_test_cornstack(u.split(":", 1)[0], snaps[s])
        c = code[h]
        hdr = header_text(u, k, c)
        if not test:
            variants["V1"][s].append((u, hdr))
            variants["V2"][s].append((u, c))
        variants["V3"][s].append((u, hdr))
    for t in tasks:
        t["gold_fail"] = list(t["gold"])
        t["gold_drop"] = [g for g in t["gold"] if kinds.get((t["snap"], g)) != "class"]
        # LocAgent labels Class.__init__ changes as Class: map class gold back to its __init__ method
        t["gold_init"] = [g + ".__init__" if kinds.get((t["snap"], g)) == "class" else g for g in t["gold"]]
    return tasks, variants


def metrics(tasks, scores, units_by_snap, gold_key):
    per = []
    for i, t in enumerate(tasks):
        gold = t[gold_key]
        if not gold:
            continue
        best = {}
        for (u, _), s in zip(units_by_snap[t["snap"]], scores[i]):
            if s > best.get(u, -1e9):
                best[u] = s
        ranked = sorted(best, key=best.get, reverse=True)
        pos = {u: j + 1 for j, u in enumerate(ranked)}
        need = max(pos.get(g, 10**9) for g in gold)
        fb = {}
        for u, s in best.items():
            f = u.split(":", 1)[0]
            fb[f] = max(fb.get(f, -1e9), s)
        fr = {f: j + 1 for j, f in enumerate(sorted(fb, key=fb.get, reverse=True))}
        fneed = max(fr.get(g.split(":", 1)[0], 10**9) for g in gold)
        per.append({"instance_id": t["instance_id"], "func_need_k": need, "file_need_k": fneed})
    acc = lambda key, k: 100 * float(np.mean([p[key] <= k for p in per]))  # noqa: E731
    return {"n": len(per), "func_acc": {k: acc("func_need_k", k) for k in (1, 5, 10, 20)},
            "file_acc": {k: acc("file_need_k", k) for k in (1, 3, 5)}, "per_instance": per}


def embed_vllm(name, texts, max_len):
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    global _LLM
    if _LLM.get("name") != name:
        if _LLM.get("llm") is not None:
            del _LLM["llm"]
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        _LLM.update(name=name, llm=LLM(model=name, revision=model_rev(name), runner="pooling", trust_remote_code=True, dtype="float16",
                                       max_model_len=8192, gpu_memory_utilization=0.85),
                    tok=AutoTokenizer.from_pretrained(name, revision=model_rev(name), trust_remote_code=True))
    ids = _LLM["tok"](texts, truncation=True, max_length=max_len)["input_ids"]
    t = time.time()
    outs = _LLM["llm"].embed([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
    E = np.array([o.outputs.embedding for o in outs], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    assert np.isfinite(E).all()
    print(f"[progress] embedded {len(texts)} in {time.time() - t:.0f}s", flush=True)
    return E


_LLM: dict = {}


def run_dense(model, tasks, variants, vs, out):
    name = HF[model]
    Q = embed_vllm(name, [QUERY_PREFIX + t["query"] for t in tasks], 1024)
    res = {}
    for v in vs:
        ubs = variants[v]
        uniq = sorted({x for s in ubs.values() for _, x in s})
        row = {x: j for j, x in enumerate(uniq)}
        print(f"[calib] {model} {v}: {len(uniq)} unique texts", flush=True)
        D = embed_vllm(name, uniq, 512)
        scores = [D[[row[x] for _, x in ubs[t["snap"]]]] @ Q[i] for i, t in enumerate(tasks)]
        for gk in ("gold_drop", "gold_fail", "gold_init"):
            m = metrics(tasks, scores, ubs, gk)
            res[f"{model}_{v}_{gk}"] = m
            print(f"[calib] {model} {v} {gk}: n={m['n']} Acc@5 {m['func_acc'][5]:.2f} Acc@10 {m['func_acc'][10]:.2f} "
                  f"file@1 {m['file_acc'][1]:.2f}", flush=True)
            (out / f"calib_{model}_{v}_{gk}.json").write_text(json.dumps(m))
    return res


def run_lateon(model, tasks, variants, vs, out):
    import torch
    from pylate import models
    m_ = models.ColBERT(model_name_or_path=HF[model], revision=model_rev(HF[model]), document_length=512, device="cuda")
    enc = lambda xs, q: [e.astype(np.float16) if isinstance(e, np.ndarray) else e.float().cpu().numpy().astype(np.float16)  # noqa: E731
                         for e in m_.encode(xs, is_query=q, batch_size=64, show_progress_bar=False)]
    Q = enc([t["query"] for t in tasks], True)
    for v in vs:
        ubs = variants[v]
        uniq = sorted({x for s in ubs.values() for _, x in s})
        row = {x: j for j, x in enumerate(uniq)}
        t0 = time.time()
        order = sorted(range(len(uniq)), key=lambda j: len(uniq[j]))
        E = [None] * len(uniq)
        for s in range(0, len(order), 4096):
            idx = order[s:s + 4096]
            for j, e in zip(idx, enc([uniq[j] for j in idx], False)):
                E[j] = e
            print(f"[progress] {model} {v} {min(len(order), s + 4096)}/{len(order)} {time.time() - t0:.0f}s", flush=True)
        scores = []
        for i, t in enumerate(tasks):
            docs = [E[row[x]] for _, x in ubs[t["snap"]]]
            q = torch.from_numpy(Q[i]).cuda().float()
            sc = np.zeros(len(docs), np.float32)
            o = sorted(range(len(docs)), key=lambda j: docs[j].shape[0])
            for s in range(0, len(o), 1024):
                idx = o[s:s + 1024]
                L = max(d.shape[0] for d in (docs[j] for j in idx))
                Dm = torch.zeros(len(idx), L, q.shape[1]); M = torch.zeros(len(idx), L, dtype=torch.bool)
                for r, j in enumerate(idx):
                    Dm[r, :docs[j].shape[0]] = torch.from_numpy(docs[j].astype(np.float32)); M[r, :docs[j].shape[0]] = True
                Dm, M = Dm.cuda(), M.cuda()
                sim = torch.einsum("qd,bld->bql", q, Dm).masked_fill(~M[:, None, :], float("-inf")).amax(-1).sum(-1)
                sc[idx] = sim.cpu().numpy()
            scores.append(sc)
        for gk in ("gold_drop", "gold_fail", "gold_init"):
            m = metrics(tasks, scores, ubs, gk)
            print(f"[calib] {model} {v} {gk}: n={m['n']} Acc@5 {m['func_acc'][5]:.2f} Acc@10 {m['func_acc'][10]:.2f} "
                  f"file@1 {m['file_acc'][1]:.2f}", flush=True)
            (out / f"calib_{model}_{v}_{gk}.json").write_text(json.dumps(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--variants", default="V1,V2,V3")
    ap.add_argument("--out", default="/content/out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks, variants = build(Path(a.data))
    stats = {v: {"units": sum(len(x) for x in d.values()), "unique": len({t for x in d.values() for _, t in x})}
             for v, d in variants.items()}
    stats["class_gold_instances"] = sum(len(t["gold_drop"]) < len(t["gold_fail"]) for t in tasks)
    stats["no_function_gold_instances"] = sum(not t["gold_drop"] for t in tasks)
    print("[calib] corpus stats", json.dumps(stats), flush=True)
    (out / "calib_stats.json").write_text(json.dumps(stats))
    vs = a.variants.split(",")
    for m in a.models.split(","):
        (run_lateon if m.startswith("LATEON") else run_dense)(m, tasks, variants, vs, out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # vLLM engine processes can keep the interpreter alive after main() returns (cost 5 CU once)
