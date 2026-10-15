"""Subprocess worker: embed texts with vLLM (pooling runner). Inputs/outputs are files so the parent never
imports vLLM. The process always ends with os._exit (vLLM engines can keep the interpreter alive).
Usage: python colab/vllm_embed.py <model_dir_or_hub_id> <texts.pkl> <max_len> <out.npy>"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (issueloc)


def main():
    name, src, max_len, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(src)
    n = pf.metadata.num_rows
    from issueloc.revisions import model_rev
    tok = AutoTokenizer.from_pretrained(name, revision=model_rev(name), trust_remote_code=True)
    dtype = "bfloat16" if "Qwen3-Embedding" in name else "float16"
    llm = LLM(model=name, revision=model_rev(name), runner="pooling", trust_remote_code=True, dtype=dtype, max_model_len=8192,
              gpu_memory_utilization=0.85)
    out = None
    t0, s = time.time(), 0
    for batch in pf.iter_batches(batch_size=50000, columns=["text"]):
        chunk = batch.column(0).to_pylist()
        ids = tok(chunk, truncation=True, max_length=max_len)["input_ids"]
        res = llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
        E = np.array([r.outputs.embedding for r in res], dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        if not np.isfinite(E).all():
            raise FloatingPointError("non-finite embeddings")
        if out is None:
            out = np.lib.format.open_memmap(dst + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n, E.shape[1]))
        out[s:s + len(E)] = E.astype(np.float16)
        s += len(E)
        el = time.time() - t0
        print(f"[progress] vllm {os.path.basename(name)} {s}/{n} {el:.0f}s eta {el / s * (n - s):.0f}s", flush=True)
    assert s == n
    out.flush(); del out
    os.replace(dst + ".tmp.npy", dst)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except BaseException as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
