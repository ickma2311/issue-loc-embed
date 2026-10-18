"""Turn a training checkpoint (M2.pt from gpu_run.py train) into a sentence-transformers model directory like the
release: config, tokenizer and remote code copied from an existing release directory; model.safetensors keeps only
the base model's 112 tensors. Also writes a flat copy for a Kaggle dataset (v02__<path with __>), a SHA256SUMS file,
and copies of the evaluated SWE-Bench-Lite files. Applied to v0.1's M2.pt it reproduces the v0.1 release file.
Usage: python scripts/make_model_dir.py <M2.pt> <release dir> <dir with eval_*_paper files> <out dir>
"""
import hashlib, shutil, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LITE = {"eval_tasks_paper.parquet": "9af40fae", "eval_ent_paper_000.parquet": "6b5709f3", "eval_txt_paper_000.parquet": "09f7deea"}
KEEP = ["config.json", "config_sentence_transformers.json", "configuration_hf_nomic_bert.py", "modeling_hf_nomic_bert.py",
        "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "1_Pooling/config.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(pt, rel, paper, out):
    rel, out = Path(rel), Path(out)
    d = out / "issue-loc-embed-v02"; flat = out / "kaggle_input"
    shutil.rmtree(out, ignore_errors=True); (d / "1_Pooling").mkdir(parents=True); flat.mkdir(parents=True)
    for f in KEEP:
        shutil.copy(rel / f, d / f)
    base = load_file(rel / "model.safetensors")            # key names and dtypes of the v0.1 release (112 tensors)
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    norm = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    new = {}
    for k, v in base.items():
        src = norm.get(k, norm.get(k[len("model."):]) if k.startswith("model.") else None)
        assert src is not None, f"missing tensor {k}"
        assert tuple(src.shape) == tuple(v.shape), (k, src.shape, v.shape)
        new[k] = src.to(v.dtype).contiguous()
    save_file(new, d / "model.safetensors", metadata={"format": "pt"})
    files = sorted(p for p in d.rglob("*") if p.is_file())
    lines = []
    for p in files:
        rp = p.relative_to(d).as_posix()
        shutil.copy(p, flat / ("v02__" + rp.replace("/", "__")))
        lines.append(f"{sha(p)}  {rp}")
    (flat / "V02_SHA256SUMS").write_text("\n".join(lines) + "\n")
    for f, pre in LITE.items():
        assert sha(Path(paper) / f).startswith(pre), f
        shutil.copy(Path(paper) / f, flat / f)
    print("model.safetensors", sha(d / "model.safetensors"), "tensors", len(new), "M2.pt", sha(pt))


if __name__ == "__main__":
    main(*sys.argv[1:])
