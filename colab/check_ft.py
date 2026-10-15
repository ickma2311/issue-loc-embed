"""Before serving a fine-tuned checkpoint with vLLM, check vLLM == HF on 64 pool texts (cos >= 0.999).
Usage: python colab/check_ft.py <state_dict.pt> <texts.pkl>   -> exit 0 ok / 3 mismatch"""
import os, pickle, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from issueloc.model import BASE, encode, export_hf_dir, load_encoder

pt, src = sys.argv[1], sys.argv[2]
texts = pickle.load(open(src, "rb"))[:64]
tok, m = load_encoder(BASE, "cuda", pt)
hf = np.zeros((len(texts), 768), np.float16)
encode(tok, m, texts.__getitem__, list(range(len(texts))), 256, hf)
del m
import torch; torch.cuda.empty_cache()
d = tempfile.mkdtemp()
import pyarrow as pa, pyarrow.parquet as pq
pq.write_table(pa.table({"text": texts}), f"{d}/t.parquet")
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "vllm_embed.py"), export_hf_dir(pt), f"{d}/t.parquet", "256", f"{d}/e.npy"],
                   timeout=1200, env=dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0"))
vl = np.load(f"{d}/e.npy")
cos = (hf.astype(np.float32) * vl.astype(np.float32)).sum(1)
print(f"[check_ft] {pt}: cos min {cos.min():.5f} mean {cos.mean():.5f}", flush=True)
sys.exit(0 if r.returncode == 0 and cos.min() >= 0.999 else 3)
