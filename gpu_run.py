"""GPU stages: calibration, mining, training and evaluation.

  calibrate  evaluate B0 (CodeRankEmbed) and REF (SweRankEmbed-Small) with representations R1/R2;
             the representation is the one where B0 scores higher (REF is reported only) -> rep.json
  mine       embed the training pool with a model, consistency-filter (scaled K) and mine 15 hard
             negatives                                               -> inst_<name>.parquet
  train      InfoNCE + GradCache from CodeRankEmbed                   -> <name>.pt (fp16 state dict)
  eval       SWE-Bench-Lite function/file Acc@k                       -> eval_<name>.json

REF is used for evaluation only: never for training, filtering or mining.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from issueloc.model import BASE, QUERY_PREFIX, encode_multi_gpu, load_encoder, train_infonce
from issueloc.patch import related

REF = "Salesforce/SweRankEmbed-Small"
PAPER = {"B0": 58.76, "REF": 74.45}
K_PAPER, N_NEG = 20, 15


def log(msg):
    print(f"[gpu] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_data(root: str) -> Path:
    fname = "train_tasks_paper.parquet" if os.environ.get("EVAL_SET", "").startswith("paper") else "train_tasks.parquet"
    hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
    if not hits:
        raise SystemExit(f"no data/train_tasks.parquet under {root}")
    return Path(hits[0]).parent


def rep_text(rep: str, path: str, text: str) -> str:
    if EVAL_SET.startswith("paper") and not path:  # paper-format texts are final (header or code-only variant)
        return text
    return text if rep == "R1" else f"{path}\n{text}"


QWEN06 = "Qwen/Qwen3-Embedding-0.6B"
QWEN_PREFIX = "Instruct: Given a GitHub issue, retrieve the code that needs to be changed to resolve it\nQuery:"


def query_prefix(m: str) -> str:
    return QWEN_PREFIX if m == "QWEN06" else QUERY_PREFIX


def model_spec(m: str):
    if m == "B0":
        return BASE, None
    if m == "REF":
        return REF, None
    if m == "QWEN06":
        return QWEN06, None
    return BASE, m  # path to a fine-tuned state dict


# ----------------------------------------------------------------------------- eval
EVAL_SET = os.environ.get("EVAL_SET", "ours")  # "ours" (build_data.py format) or "paper" (CoRNStack/SweRank protocol)


def _sfx(kind: str = "train"):
    """File suffix. Training files are always the V1 paper files; eval may use a robustness variant."""
    if EVAL_SET.startswith("paper"):
        return "_" + EVAL_SET if kind == "eval" else "_paper"
    return ""


def eval_dir(data: Path) -> Path:
    if EVAL_SET != "paper":
        return data
    if (data / f"eval_tasks{_sfx('eval')}.parquet").exists():
        return data
    hits = glob.glob("/kaggle/input/**/eval_tasks_paper.parquet", recursive=True) + \
        glob.glob(str(data.parent / "**" / "eval_tasks_paper.parquet"), recursive=True)
    assert hits, "paper eval set not mounted"
    return Path(hits[0]).parent


def fingerprint(data: Path) -> str:
    d, x = eval_dir(data), _sfx("eval")
    h = hashlib.sha256()
    for f in (f"eval_tasks{x}.parquet", f"eval_ent{x}_000.parquet", f"eval_txt{x}_000.parquet"):
        h.update((d / f).read_bytes())
    return h.hexdigest()[:16]


def load_eval(data: Path, limit: int = 0):
    d, x = eval_dir(data), _sfx("eval")
    tasks = pq.read_table(d / f"eval_tasks{x}.parquet").to_pylist()
    tasks.sort(key=lambda t: t["instance_id"])
    if limit:
        tasks = tasks[:limit]
    else:
        assert len(tasks) == 274, f"eval set has {len(tasks)} instances, expected 274"
    assert all(len(t["gold"]) > 0 for t in tasks)
    ent = pq.read_table(d / f"eval_ent{x}_000.parquet").to_pydict()
    txt = pq.read_table(d / f"eval_txt{x}_000.parquet").to_pydict()
    return tasks, ent, txt


def run_eval(model: str, rep: str, data: Path, out: Path, name: str, limit: int = 0, models=None) -> dict:
    tasks, ent, txt = load_eval(data, limit)
    snaps = {t["snap"] for t in tasks}
    by_snap = defaultdict(list)
    for s, u, h in zip(ent["snap"], ent["uid"], ent["h"]):
        if s in snaps:
            by_snap[s].append((u, h))
    need = sorted({h for v in by_snap.values() for _, h in v})
    hpos = {h: i for i, h in enumerate(txt["h"])}
    assert rep in ("R1", "R2"), rep
    docs = lambda i: rep_text(rep, txt["path"][hpos[need[i]]], txt["text"][hpos[need[i]]])  # noqa: E731
    name_, sd = model_spec(model)
    log(f"eval {name}: {len(tasks)} tasks, {len(need)} unique units")
    D = encode_multi_gpu(name_, sd, docs, 512, progress=f"eval-docs {name}", models=models, n=len(need))
    Q = encode_multi_gpu(name_, sd, [query_prefix(model) + t["query"] for t in tasks], 1024, models=models)
    row = {h: i for i, h in enumerate(need)}

    def score(i, units):
        return D[[row[h] for _, h in units]].astype(np.float32) @ Q[i].astype(np.float32)

    return rank_and_save(tasks, by_snap, score, data, out, name, model, rep)


def eval_units(data: Path, limit: int = 0):
    """Eval tasks, {snap: [(uid, h)]}, sorted unique hashes, text table."""
    tasks, ent, txt = load_eval(data, limit)
    snaps = {t["snap"] for t in tasks}
    by_snap = defaultdict(list)
    for s, u, h in zip(ent["snap"], ent["uid"], ent["h"]):
        if s in snaps:
            by_snap[s].append((u, h))
    return tasks, by_snap, txt


def rank_and_save(tasks, by_snap, score, data: Path, out: Path, name: str, model: str, rep: str) -> dict:
    """score(task_index, units) -> similarity per unit. Shared by every evaluated model."""
    per, ks = [], (1, 3, 5, 10, 20)
    for i, t in enumerate(tasks):
        units = by_snap[t["snap"]]
        sims = np.asarray(score(i, units), dtype=np.float32)
        assert np.isfinite(sims).all() and len(sims) == len(units)
        best = {}
        for (u, _), s in zip(units, sims):
            if s > best.get(u, -9):
                best[u] = s
        ranked = sorted(best, key=best.get, reverse=True)
        pos = {u: i + 1 for i, u in enumerate(ranked)}
        gold = list(t["gold"])
        need_k = max(pos.get(g, 10**9) for g in gold)
        fbest = {}
        for u, s in best.items():
            f = u.split(":", 1)[0]
            fbest[f] = max(fbest.get(f, -9), s)
        franked = sorted(fbest, key=fbest.get, reverse=True)
        fpos = {f: i + 1 for i, f in enumerate(franked)}
        gfiles = {g.split(":", 1)[0] for g in gold}
        fneed = max(fpos.get(f, 10**9) for f in gfiles)
        per.append({"instance_id": t["instance_id"], "func_need_k": int(need_k), "file_need_k": int(fneed),
                    "gold_missing": int(sum(g not in pos for g in gold)), "n_units": len(ranked)})
    res = {"name": name, "model": model, "rep": rep, "eval_set": EVAL_SET, "n": len(per),
           "data_fingerprint": fingerprint(data),
           "func_acc": {k: 100 * float(np.mean([p["func_need_k"] <= k for p in per])) for k in ks},
           "file_acc": {k: 100 * float(np.mean([p["file_need_k"] <= k for p in per])) for k in (1, 3, 5)},
           "gold_missing_instances": int(sum(p["gold_missing"] > 0 for p in per)), "per_instance": per}
    (out / f"eval_{name}.json").write_text(json.dumps(res, indent=1))
    log(f"eval {name}: func Acc@5 {res['func_acc'][5]:.2f} Acc@10 {res['func_acc'][10]:.2f} "
        f"file Acc@1 {res['file_acc'][1]:.2f} (gold missing in {res['gold_missing_instances']})")
    return res


def calibrate(data: Path, out: Path, limit: int = 0):
    import torch
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    results = {}
    for m in ("B0", "REF"):
        models = [load_encoder(model_spec(m)[0], d) for d in devs]
        for rep in ("R1", "R2"):
            results[(m, rep)] = run_eval(m, rep, data, out, f"{m}_{rep}", limit, models=models)["func_acc"][10]
        del models
        torch.cuda.empty_cache()
    # the representation is the one where the baseline B0 is
    # strongest, so M2 is compared with B0 in its best format and REF never influences training.
    rep = "R2" if results[("B0", "R2")] > results[("B0", "R1")] else "R1"
    dev = {r: abs(results[("B0", r)] - PAPER["B0"]) + abs(results[("REF", r)] - PAPER["REF"]) for r in ("R1", "R2")}
    info = {"rep": rep, "rule": "argmax B0 Acc@10 (tie -> R1)", "deviation_from_paper": dev,
            "acc10": {f"{m}_{r}": v for (m, r), v in results.items()}, "paper": PAPER}
    (out / "rep.json").write_text(json.dumps(info, indent=1))
    log(f"calibration: {info}")
    return rep


# ----------------------------------------------------------------------------- mine
def load_train(data: Path, limit_tasks: int = 0):
    tasks = pq.read_table(data / f"train_tasks{_sfx()}.parquet").to_pylist()
    rep_f = data / "report.json"
    if rep_f.exists():  # repos sharing a project name with an eval repo (likely renamed copies): drop
        drop = set(json.loads(rep_f.read_text()).get("fork_like_train_repos", []))
        n0 = len(tasks)
        tasks = [t for t in tasks if t["repo"] not in drop]
        log(f"dropped {n0 - len(tasks)} tasks from {len(drop)} repos sharing an eval project name")
    tasks.sort(key=lambda t: t["instance_id"])
    if limit_tasks:
        rng = random.Random(0)
        tasks = rng.sample(tasks, min(limit_tasks, len(tasks)))
    snaps = {t["snap"] for t in tasks}
    by_snap = defaultdict(list)
    for f in sorted(glob.glob(str(data / f"train_ent{_sfx()}_*.parquet"))):
        e = pq.read_table(f).to_pydict()
        for s, u, h in zip(e["snap"], e["uid"], e["h"]):
            if s in snaps:
                by_snap[s].append((u, h))
    need = {h for v in by_snap.values() for _, h in v}
    texts = {}
    for f in sorted(glob.glob(str(data / f"train_txt{_sfx()}_*.parquet"))):
        t = pq.read_table(f).to_pydict()
        for h, p, x in zip(t["h"], t["path"], t["text"]):
            if h in need and h not in texts:
                texts[h] = (p, x)
    return tasks, by_snap, texts


def k_eff(pool_n: int, full_est: int) -> int:
    """Paper keeps an instance if its positive ranks in the top 20 of the whole repository.
    Our pool is a file-level sample of the repository, so the threshold is scaled by pool/full."""
    return int(min(K_PAPER, max(1, round(K_PAPER * pool_n / max(pool_n, full_est)))))


def load_train_lowmem(data: Path, limit_tasks: int = 0):
    """Like load_train but keeps the pool as arrays and never loads unit texts."""
    import pyarrow as pa
    import pyarrow.compute as pc
    tasks = pq.read_table(data / f"train_tasks{_sfx()}.parquet").to_pylist()
    tasks.sort(key=lambda t: t["instance_id"])
    if limit_tasks:
        tasks = random.Random(0).sample(tasks, min(limit_tasks, len(tasks)))
    ent = pa.concat_tables([pq.read_table(f, columns=["snap", "uid", "h"])
                            for f in sorted(glob.glob(str(data / f"train_ent{_sfx()}_*.parquet")))])
    ent = ent.filter(pc.is_in(ent["snap"], value_set=pa.array(sorted({t["snap"] for t in tasks}))))
    ent = ent.sort_by("snap")
    snap = ent["snap"].combine_chunks()
    uid = ent["uid"].combine_chunks()
    h = ent["h"].combine_chunks().to_numpy()
    s_list = snap.to_pylist()
    offs, start = {}, 0
    for i in range(1, len(s_list) + 1):
        if i == len(s_list) or s_list[i] != s_list[start]:
            offs[s_list[start]] = (start, i); start = i
    del s_list
    return tasks, offs, uid, h


def mine(model: str, rep: str, data: Path, out: Path, name: str, keep_from: str | None = None,
         limit_tasks: int = 0):
    assert rep in ("R1", "R2"), rep
    assert model != "REF", "the NC reference model is evaluation-only"
    if os.environ.get("EMBED_BACKEND") == "vllm" and EVAL_SET.startswith("paper"):
        return mine_lowmem(model, data, out, name, keep_from, limit_tasks)
    tasks, by_snap, texts = load_train(data, limit_tasks)
    keep = None
    if keep_from:
        keep = set(json.loads(Path(keep_from).read_text())["kept"])
        snaps = {t["snap"] for t in tasks if any(f"{t['instance_id']}|{p}" in keep for p in t["positives"])}
        tasks = [t for t in tasks if t["snap"] in snaps]
    hs = sorted({h for t in tasks for _, h in by_snap[t["snap"]]})
    row = {h: i for i, h in enumerate(hs)}
    log(f"mine {name}: {len(tasks)} tasks, {len(hs)} unique pool units "
        f"(~{len(hs) * 768 * 2 / 1e9:.1f} GB embeddings)")
    name_, sd = model_spec(model)
    import torch
    devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    models = None if os.environ.get("EMBED_BACKEND") == "vllm" else [load_encoder(name_, d, sd) for d in devs]
    D = encode_multi_gpu(name_, sd, lambda i: rep_text(rep, *texts[hs[i]]), 256, progress=f"pool {name}",
                         models=models, n=len(hs))
    Q = encode_multi_gpu(name_, sd, [QUERY_PREFIX + t["query"] for t in tasks], 512, models=models)
    del models
    filtered, unfiltered, kept_ids, stats = [], [], [], defaultdict(int)
    for t, q in zip(tasks, Q):
        units = by_snap[t["snap"]]
        sims = D[[row[h] for _, h in units]].astype(np.float32) @ q.astype(np.float32)
        best, best_h = {}, {}
        for (u, h), s in zip(units, sims):
            if s > best.get(u, -9):
                best[u], best_h[u] = s, h
        pos = [p for p in t["positives"] if p in best]
        allpos = t.get("excl_positives") or t.get("all_positives") or t["positives"]
        excl = {u for u in best if any(related(u, p) for p in allpos)}
        cand = sorted((u for u in best if u not in excl), key=best.get, reverse=True)
        if not cand:
            stats["no_negatives"] += 1
            continue
        negs = cand[:N_NEG]
        negs = (negs * (N_NEG // len(negs) + 1))[:N_NEG]
        neg_txt = [rep_text(rep, *texts[best_h[u]]) for u in negs]
        K = k_eff(t["pool_n"], t["full_est"])
        cand_s = np.array([best[u] for u in cand], dtype=np.float32)
        for p in pos:
            rank = int((cand_s > best[p]).sum()) + 1
            inst = {"id": f"{t['instance_id']}|{p}", "q": t["query"], "pos": rep_text(rep, *texts[best_h[p]]),
                    "negs": neg_txt, "rank": rank, "K": K}
            unfiltered.append(inst)
            stats["instances"] += 1
            if keep is not None:
                if inst["id"] in keep:
                    filtered.append(inst)
            elif rank <= K:
                filtered.append(inst)
                kept_ids.append(inst["id"])
    stats["filtered"] = len(filtered)
    rng = random.Random(0)
    nof = rng.sample(unfiltered, min(len(unfiltered), len(filtered))) if keep is None else []
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(filtered), out / f"inst_{name}.parquet", compression="zstd")
    if keep is None:
        pq.write_table(pa.Table.from_pylist(nof), out / f"inst_{name}_nofilter.parquet", compression="zstd")
        (out / f"kept_{name}.json").write_text(json.dumps({"kept": kept_ids}))
    ks = [x["K"] for x in unfiltered]
    hist = np.histogram(ks, bins=[1, 2, 3, 5, 10, 20, 21])[0].tolist() if ks else []
    stats.update(K_mean=float(np.mean(ks)) if ks else 0, K_eq_20=int(sum(k == 20 for k in ks)),
                 K_hist_bins_1_2_3_5_10_20=hist, retention=len(filtered) / max(1, len(unfiltered)),
                 tasks=len(tasks), pool_units=len(hs))
    (out / f"mine_{name}.json").write_text(json.dumps(stats, indent=1))
    log(f"mine {name}: {dict(stats)}")


def mine_lowmem(model: str, data: Path, out: Path, name: str, keep_from: str | None, limit_tasks: int):
    """Same selection rules as mine(); memory-lean for the 1.45M-unit pool (Colab)."""
    import pyarrow as pa
    tasks, offs, uid, harr = load_train_lowmem(data, limit_tasks)
    keep = None
    if keep_from:
        keep = set(json.loads(Path(keep_from).read_text())["kept"])
        snaps = {t["snap"] for t in tasks if any(f"{t['instance_id']}|{p}" in keep for p in t["positives"])}
        tasks = [t for t in tasks if t["snap"] in snaps]
    txt_files = sorted(glob.glob(str(data / f"train_txt{_sfx()}_*.parquet")))
    assert len(txt_files) == 1, txt_files
    txt_h = pq.read_table(txt_files[0], columns=["h"]).column("h").to_numpy()
    order = np.argsort(txt_h); txt_sorted = txt_h[order]
    log(f"mine {name} (low-mem): {len(tasks)} tasks, {len(txt_h)} pool texts (~{len(txt_h) * 768 * 2 / 1e9:.1f} GB embeddings)")
    name_, sd = model_spec(model)
    from issueloc.model import encode_parquet_vllm
    D = encode_parquet_vllm(name_, sd, txt_files[0], 256)
    assert D.shape[0] == len(txt_h)
    Q = encode_multi_gpu(name_, sd, [QUERY_PREFIX + t["query"] for t in tasks], 512)

    def rows_of(hh):
        j = np.searchsorted(txt_sorted, hh)
        assert (txt_sorted[j] == hh).all()
        return order[j]

    recs, stats, kept_ids = [], defaultdict(int), []
    for t, q in zip(tasks, Q):
        a, b = offs[t["snap"]]
        units = uid.slice(a, b - a).to_pylist()
        hh = harr[a:b]
        sims = D[rows_of(hh)].astype(np.float32) @ q.astype(np.float32)
        best, best_h = {}, {}
        for u, hv, s in zip(units, hh, sims):
            if s > best.get(u, -9):
                best[u], best_h[u] = s, int(hv)
        pos = [p for p in t["positives"] if p in best]
        allpos = t.get("excl_positives") or t.get("all_positives") or t["positives"]
        excl = {u for u in best if any(related(u, p) for p in allpos)}
        cand = sorted((u for u in best if u not in excl), key=best.get, reverse=True)
        if not cand:
            stats["no_negatives"] += 1
            continue
        negs = cand[:N_NEG]
        negs = (negs * (N_NEG // len(negs) + 1))[:N_NEG]
        neg_h = [best_h[u] for u in negs]
        K = k_eff(t["pool_n"], t["full_est"])
        cand_s = np.array([best[u] for u in cand], dtype=np.float32)
        for p in pos:
            rank = int((cand_s > best[p]).sum()) + 1
            iid = f"{t['instance_id']}|{p}"
            sel = (iid in keep) if keep is not None else (rank <= K)
            recs.append((iid, t["query"], best_h[p], neg_h, rank, K, sel))
            stats["instances"] += 1
            if keep is None and sel:
                kept_ids.append(iid)
    del D
    filtered = [r for r in recs if r[6]]
    rng = random.Random(0)
    nof = rng.sample(recs, min(len(recs), len(filtered))) if keep is None else []
    need = {r[2] for r in filtered + nof} | {x for r in filtered + nof for x in r[3]}
    text_of = {}
    for batch in pq.ParquetFile(txt_files[0]).iter_batches(batch_size=100000, columns=["h", "text"]):
        for hv, tx in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if hv in need:
                text_of[hv] = tx

    def inst(r):
        return {"id": r[0], "q": r[1], "pos": text_of[r[2]], "negs": [text_of[x] for x in r[3]], "rank": r[4], "K": r[5]}

    pq.write_table(pa.Table.from_pylist([inst(r) for r in filtered]), out / f"inst_{name}.parquet", compression="zstd")
    if keep is None:
        pq.write_table(pa.Table.from_pylist([inst(r) for r in nof]), out / f"inst_{name}_nofilter.parquet", compression="zstd")
        (out / f"kept_{name}.json").write_text(json.dumps({"kept": kept_ids}))
    ks = [r[5] for r in recs]
    hist = np.histogram(ks, bins=[1, 2, 3, 5, 10, 20, 21])[0].tolist() if ks else []
    stats.update(filtered=len(filtered), K_mean=float(np.mean(ks)) if ks else 0, K_eq_20=int(sum(k == 20 for k in ks)),
                 K_hist_bins_1_2_3_5_10_20=hist, retention=len(filtered) / max(1, len(recs)),
                 tasks=len(tasks), pool_units=len(txt_h))
    (out / f"mine_{name}.json").write_text(json.dumps(stats, indent=1))
    log(f"mine {name}: {dict(stats)}")


# ----------------------------------------------------------------------------- train
def train(inst: str, out_pt: str, gpu: int, steps_file: str, max_min: float, role: str, max_steps: int = 0,
          bench_steps: int = 5):
    import torch
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    instances = pq.read_table(inst).to_pylist()
    batch = int(os.environ.get("TRAIN_BATCH", "64"))  # 64 = paper; smaller only in CPU smoke tests
    assert len(instances) >= batch, f"{inst}: only {len(instances)} instances"
    epoch_steps = max(1, len(instances) // batch)
    if role == "leader":
        tok, model = load_encoder(BASE, dev)
        t = {}
        train_infonce(model, tok, instances, bench_steps, on_step=lambda s, el: t.__setitem__(s, el),
                      log_every=10**9, batch=batch)
        sps = (t[bench_steps - 1] - t[0]) / (bench_steps - 1)
        steps = min(epoch_steps, int(max_min * 60 / sps))
        if max_steps:
            steps = min(steps, max_steps)
        tmp = steps_file + ".tmp"
        Path(tmp).write_text(json.dumps({"steps": steps, "batch": batch, "sec_per_step": sps, "epoch_steps": epoch_steps,
                                         "capped": steps < epoch_steps, "n_instances": len(instances),
                                         "inst": inst}))
        os.replace(tmp, steps_file)
        log(f"leader: {sps:.2f} s/step, epoch={epoch_steps} steps -> training {steps} steps")
        del model
        torch.cuda.empty_cache()
    else:
        while not Path(steps_file).exists():
            time.sleep(10)
        steps = json.loads(Path(steps_file).read_text())["steps"]
        log(f"follower: training {steps} steps on {len(instances)} instances (matched to leader)")
    tok, model = load_encoder(BASE, dev)
    losses = train_infonce(model, tok, instances, steps, batch=batch)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, out_pt)
    Path(out_pt + ".json").write_text(json.dumps({"steps": steps, "loss_first10": float(np.mean(losses[:10])),
                                                   "loss_last10": float(np.mean(losses[-10:]))}))
    log(f"saved {out_pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "mine", "train", "eval"])
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    ap.add_argument("--model", default="B0")
    ap.add_argument("--rep", default=None)
    ap.add_argument("--name", default="x")
    ap.add_argument("--keep-from")
    ap.add_argument("--inst")
    ap.add_argument("--out-pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps-file")
    ap.add_argument("--role", default="leader")
    ap.add_argument("--max-min", type=float, default=90)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--bench-steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = find_data(a.data_root)
    rep = a.rep or (json.loads((out / "rep.json").read_text())["rep"] if (out / "rep.json").exists() else None)
    if a.stage == "calibrate":
        calibrate(data, out, a.limit)
    elif a.stage == "mine":
        mine(a.model, rep, data, out, a.name, a.keep_from, a.limit)
    elif a.stage == "train":
        train(a.inst, a.out_pt, a.gpu, a.steps_file, a.max_min, a.role, a.max_steps, a.bench_steps)
    elif a.stage == "eval":
        run_eval(a.model, rep, data, out, a.name, a.limit)


if __name__ == "__main__":
    sys.exit(main())
