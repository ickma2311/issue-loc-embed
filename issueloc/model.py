"""Bi-encoder utilities: CLS-pooled, L2-normalised embeddings; InfoNCE training with GradCache.

Training follows SweRank §4.1 (InfoNCE over the batch's positives and all mined hard negatives)
and Appendix A (lr 2e-5, batch 64, 15 hard negatives). GradCache (Gao et al., 2021) is
re-implemented here so batch 64 x 16 documents fits a 16 GB T4.
"""
from __future__ import annotations

import math
import os
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

QUERY_PREFIX = "Represent this query for searching relevant code: "
BASE = "nomic-ai/CodeRankEmbed"


def load_encoder(name: str, device: str, state_dict_path: str | None = None):
    import os
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    # The nomic-bert remote code cannot resolve hub weights under recent transformers; loading from a
    # local snapshot takes its direct-safetensors path (strict=False), so the keys are checked here.
    from issueloc.revisions import model_rev
    local = snapshot_download(name, revision=model_rev(name))
    import json as _json
    pool = _json.load(open(os.path.join(local, "1_Pooling", "config.json")))
    assert pool.get("pooling_mode_cls_token") and not pool.get("pooling_mode_mean_tokens"), f"{name}: not CLS pooling"
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    model = AutoModel.from_pretrained(local, trust_remote_code=True)
    if not hasattr(model, "get_extended_attention_mask"):
        # removed from recent transformers; the nomic-bert remote code still calls it. Standard BERT form.
        def _gem(self, attention_mask, input_shape, dtype=None):
            dtype = dtype or next(self.parameters()).dtype
            ext = attention_mask[:, None, None, :].to(dtype)
            return (1.0 - ext) * torch.finfo(dtype).min
        type(model).get_extended_attention_mask = _gem
    ref = load_file(os.path.join(local, "model.safetensors"))
    msd = model.state_dict()
    absent = [k for k in msd if k not in ref and "pooler" not in k]
    unused = [k for k in ref if k not in msd]
    assert not absent and not unused, f"{name}: weights not loaded {absent[:5]} / unused {unused[:5]}"
    for k in list(ref)[:3] + list(ref)[-3:]:
        assert torch.allclose(msd[k].float(), ref[k].float()), f"{name}: tensor {k} differs after load"
    if state_dict_path:
        sd = torch.load(state_dict_path, map_location="cpu")
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        assert not unexpected and not [m for m in missing if "pooler" not in m], (missing, unexpected)
    return tok, model.to(device)


def _cls(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return F.normalize(h[:, 0].float(), dim=-1)


@torch.no_grad()
def encode(tok, model, get_text, idxs, max_len: int, out: np.ndarray, token_budget: int | None = None,
           progress: str | None = None) -> None:
    """Write normalised float16 embeddings of ``get_text(i)`` for i in ``idxs`` into ``out[i]``.
    Batches are length-sorted; the batch size is corrected with exact token counts."""
    model.eval()
    dev = next(model.parameters()).device
    token_budget = token_budget or int(os.environ.get("ENC_TOKENS", "65536"))
    n = len(idxs)
    if n == 0:
        return
    lens = {i: len(get_text(i)) for i in idxs}
    order = sorted(idxs, key=lens.get)
    t0, done, last, pos = time.time(), 0, 0.0, 0
    while pos < n:
        est = min(max_len, max(8, lens[order[min(n - 1, pos + 255)]] // 3 + 2))
        bs = max(1, min(1024, token_budget // est))
        while True:
            idx = order[pos:pos + bs]
            enc = tok([get_text(j) for j in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            L = enc["input_ids"].shape[1]
            if bs == 1 or len(idx) * L <= token_budget * 1.25:
                break
            bs = max(1, token_budget // L)
        enc = enc.to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            e = _cls(model, enc)
        e = e.cpu().numpy()
        if not np.isfinite(e).all():
            raise FloatingPointError("non-finite embeddings")
        out[idx] = e.astype(np.float16)
        pos += len(idx)
        done += len(idx)
        if progress and time.time() - last > 30:
            last = time.time()
            el = time.time() - t0
            print(f"[progress] {progress} {done}/{n} {el:.0f}s eta {el / done * (n - done):.0f}s", flush=True)


def export_hf_dir(state_dict_path: str, base: str = BASE) -> str:
    """Fine-tuned state dict -> a loadable HF model directory (copy of the base snapshot with new weights),
    so vLLM can serve it. Keys equal the base safetensors keys (checked in load_encoder)."""
    import shutil
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from safetensors import safe_open
    out = state_dict_path + ".hf"
    st = os.stat(state_dict_path)
    stamp = f"{st.st_size}:{st.st_mtime_ns}"
    if os.path.exists(os.path.join(out, "SOURCE")) and open(os.path.join(out, "SOURCE")).read() == stamp:
        return out
    from issueloc.revisions import model_rev
    src = snapshot_download(base, revision=model_rev(base))
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out, symlinks=False)
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        base_keys = set(f.keys())
    sd = torch.load(state_dict_path, map_location="cpu")
    missing = base_keys - set(sd)
    assert not missing, f"fine-tuned state dict lacks base keys: {sorted(missing)[:5]}"
    tmp = os.path.join(out, "model.safetensors.tmp")
    save_file({k: sd[k].float().contiguous() for k in sorted(base_keys)}, tmp)  # only base keys (no pooler)
    os.replace(tmp, os.path.join(out, "model.safetensors"))
    open(os.path.join(out, "SOURCE"), "w").write(stamp)
    return out


def _encode_vllm(name: str, state_dict_path: str | None, texts, max_len: int, n: int) -> np.ndarray:
    import pickle, subprocess, sys, tempfile
    target = export_hf_dir(state_dict_path) if state_dict_path else name
    get_text = texts if callable(texts) else texts.__getitem__
    with tempfile.TemporaryDirectory(dir=os.environ.get("VLLM_TMP", "/content/tmp") if os.path.isdir(
            os.environ.get("VLLM_TMP", "/content/tmp")) else None) as d:
        src, dst = os.path.join(d, "t.parquet"), os.path.join(d, "e.npy")
        import pyarrow as pa, pyarrow.parquet as pq
        schema = pa.schema([("text", pa.string())])
        with pq.ParquetWriter(src, schema) as w:  # streamed in batches: never a full second copy of the texts
            for s in range(0, n, 50000):
                w.write_table(pa.table({"text": [get_text(i) for i in range(s, min(n, s + 50000))]}, schema=schema))
        worker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "vllm_embed.py")
        env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0")
        from issueloc.procs import kill_session, popen_session, unregister
        p = popen_session([sys.executable, worker, target, src, str(max_len), dst], env=env)
        try:
            rc = p.wait(timeout=int(os.environ.get("VLLM_TIMEOUT", "5400")))
        except subprocess.TimeoutExpired:
            kill_session(p.pid)
            raise RuntimeError("vLLM worker timed out (process group killed)")
        finally:
            unregister(p.pid)
        if rc != 0 or not os.path.exists(dst):
            raise RuntimeError(f"vLLM worker failed (rc={rc})")
        out = np.array(np.load(dst, mmap_mode="r"))
        used = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                              capture_output=True, text=True).stdout.strip()
        print(f"[vllm] worker done; GPU memory used now {used} MiB", flush=True)
    if out.shape[0] != n or not np.isfinite(out).all():
        raise FloatingPointError("vLLM embeddings incomplete or non-finite")
    return out


def encode_parquet_vllm(name: str, state_dict_path: str | None, parquet_path: str, max_len: int) -> np.ndarray:
    """vLLM-embed the "text" column of an existing parquet file (row order preserved). The caller never
    loads the texts; used for the 1.45M-unit mining pool."""
    import subprocess, sys, tempfile
    target = export_hf_dir(state_dict_path) if state_dict_path else name
    tmpdir = os.environ.get("VLLM_TMP", "/content/tmp")
    with tempfile.TemporaryDirectory(dir=tmpdir if os.path.isdir(tmpdir) else None) as d:
        dst = os.path.join(d, "e.npy")
        worker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "vllm_embed.py")
        from issueloc.procs import kill_session, popen_session, unregister
        p = popen_session([sys.executable, worker, target, parquet_path, str(max_len), dst],
                          env=dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0"))
        try:
            rc = p.wait(timeout=int(os.environ.get("VLLM_TIMEOUT", "5400")))
        except subprocess.TimeoutExpired:
            kill_session(p.pid)
            raise RuntimeError("vLLM worker timed out (process group killed)")
        finally:
            unregister(p.pid)
        if rc != 0 or not os.path.exists(dst):
            raise RuntimeError(f"vLLM worker failed (rc={rc})")
        out = np.array(np.load(dst, mmap_mode="r"))
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite embeddings")
    return out


def encode_multi_gpu(name: str, state_dict_path: str | None, texts, max_len: int,
                     progress: str | None = None, models=None, n: int | None = None) -> np.ndarray:
    """Split across all visible GPUs (one model copy and one thread per GPU), or run the vLLM worker when
    EMBED_BACKEND=vllm. ``texts`` is a list or a callable i -> str (then ``n`` is required)."""
    get_text = texts if callable(texts) else texts.__getitem__
    n = len(texts) if n is None else n
    if os.environ.get("EMBED_BACKEND") == "vllm":
        return _encode_vllm(name, state_dict_path, texts, max_len, n)
    if models is None:
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
        models = [load_encoder(name, d, state_dict_path) for d in devs]
    out = np.full((n, 768), np.nan, dtype=np.float16)
    parts = np.array_split(np.arange(n), len(models))
    errors = [None] * len(models)

    def work(k):
        try:
            tok, m = models[k]
            encode(tok, m, get_text, parts[k].tolist(), max_len, out,
                   progress=(f"{progress}[gpu{k}]" if progress else None))
        except BaseException as e:  # re-raised in the main thread
            errors[k] = e

    th = [threading.Thread(target=work, args=(k,)) for k in range(len(models))]
    [t.start() for t in th]
    [t.join() for t in th]
    for e in errors:
        if e is not None:
            raise e
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{int((~np.isfinite(out).all(1)).sum())} rows not embedded")
    return out


class _RandState:
    """Capture / replay CPU+CUDA RNG so the GradCache re-forward sees identical dropout masks."""

    def __init__(self, device):
        self.device = device
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def restore(self):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, self.device)


def train_infonce(model, tok, instances: list[dict], steps: int, *, lr=2e-5, batch=64, n_neg=15,
                  q_len=512, d_len=256, temp=0.05, warmup_frac=0.05, seed=0, q_chunk=16, d_chunk=64,
                  log_every=10, on_step=None):
    """instances: [{"q": str, "pos": str, "negs": [str]*n_neg}]. Trains in place; returns loss log.
    Docs in the denominator: every positive and every hard negative in the batch (SweRank eq. 1)."""
    assert len(instances) >= batch, f"only {len(instances)} instances for batch {batch}"
    dev = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    order = list(range(len(instances)))
    rng.shuffle(order)
    need = steps * batch
    while len(order) < need:  # more than one epoch requested (not used in our runs)
        extra = list(range(len(instances)))
        rng.shuffle(extra)
        order += extra
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    model.train()
    log, t0 = [], time.time()
    for step in range(steps):
        ids = order[step * batch:(step + 1) * batch]
        qs = [QUERY_PREFIX + instances[i]["q"] for i in ids]
        docs = []
        for i in ids:
            docs.append(instances[i]["pos"])
            docs.extend(instances[i]["negs"][:n_neg])
        labels = torch.arange(len(ids), device=dev) * (1 + n_neg)
        # in-batch false negatives: a column whose text equals the row's positive (other than the label)
        dup = torch.zeros(len(ids), len(docs), dtype=torch.bool, device=dev)
        for r, i in enumerate(ids):
            ptxt = instances[i]["pos"]
            for c, dtxt in enumerate(docs):
                if c != r * (1 + n_neg) and dtxt == ptxt:
                    dup[r, c] = True

        def chunks(texts, max_len, size):
            enc = []
            for s in range(0, len(texts), size):
                enc.append(tok(texts[s:s + size], padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt").to(dev))
            return enc

        q_enc, d_enc = chunks(qs, q_len, q_chunk), chunks(docs, d_len, d_chunk)
        # 1) no-grad forward, remembering RNG state per chunk
        q_states, d_states, q_embs, d_embs = [], [], [], []
        with torch.no_grad():
            for enc, states, embs in ((q_enc, q_states, q_embs), (d_enc, d_states, d_embs)):
                for b in enc:
                    states.append(_RandState(dev))
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                        embs.append(_cls(model, b))
        q_all = torch.cat(q_embs).detach().requires_grad_()
        d_all = torch.cat(d_embs).detach().requires_grad_()
        # 2) loss on cached embeddings
        logits = (q_all @ d_all.T / temp).masked_fill(dup, -1e4)
        loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        gq, gd = q_all.grad.split([b["input_ids"].shape[0] for b in q_enc]), \
            d_all.grad.split([b["input_ids"].shape[0] for b in d_enc])
        # 3) re-forward each chunk with grad and push the cached embedding gradients through
        for enc, states, grads in ((q_enc, q_states, gq), (d_enc, d_states, gd)):
            for b, st, g in zip(enc, states, grads):
                cur = _RandState(dev)
                st.restore()
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    e = _cls(model, b)
                (e * g).sum().backward()
                cur.restore()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append(float(loss))
        if on_step is not None:
            on_step(step, time.time() - t0)
        if step % log_every == 0 or step == steps - 1:
            el = time.time() - t0
            print(f"[progress] train {step + 1}/{steps} loss={np.mean(log[-log_every:]):.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {el:.0f}s eta {el / (step + 1) * (steps - step - 1):.0f}s",
                  flush=True)
    model.eval()
    return log


def sec(x: float) -> str:
    return f"{x / 60:.1f} min" if x < 7200 else f"{x / 3600:.2f} h"


def ceil_div(a, b):
    return int(math.ceil(a / b))
