"""General code-search regression check: MTEB CodeSearchNetRetrieval, Python split
(1,000 docstring queries vs 1,000 functions; MIT). Reports nDCG@10, MRR@10, Acc@1 for one model.
Usage: python csn_eval.py --model B0|<state_dict.pt> --name NAME [--out DIR]"""
import argparse, json, os, sys, time
from pathlib import Path
import fsspec
import numpy as np
import pyarrow.parquet as pq

from gpu_run import log, model_spec, query_prefix
from issueloc.model import encode_multi_gpu

from issueloc.revisions import DATASET_PARQUET_REV
URL = ("https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval/resolve/"
       + DATASET_PARQUET_REV["mteb/CodeSearchNetRetrieval"] + "/python-{}/test/0000.parquet")


def read(part):
    for a in range(4):
        try:
            with fsspec.open(URL.format(part), "rb") as f:
                return pq.read_table(f).to_pylist()
        except Exception as e:  # transient HF errors
            log(f"retry {part}: {e}"); time.sleep(10 * (a + 1))
    raise SystemExit(f"cannot read {part}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "out"))
    a = ap.parse_args()
    corpus, queries, qrels = read("corpus"), read("queries"), read("qrels")
    cid = [c["id"] for c in corpus]
    docs = [((c.get("title") or "") + " " + c["text"]).strip() for c in corpus]
    rel = {}
    for r in qrels:
        if r["score"] > 0:
            rel.setdefault(r["query-id"], set()).add(r["corpus-id"])
    qs = [q for q in queries if q["id"] in rel]
    name_, sd = model_spec(a.model)
    D = encode_multi_gpu(name_, sd, docs, 512).astype(np.float32)
    Q = encode_multi_gpu(name_, sd, [query_prefix(a.model) + q["text"] for q in qs], 512).astype(np.float32)
    S = Q @ D.T
    ndcg, mrr, acc1 = [], [], []
    for i, q in enumerate(qs):
        order = np.argsort(-S[i])[:10]
        hits = [cid[j] in rel[q["id"]] for j in order]
        dcg = sum(1 / np.log2(r + 2) for r, h in enumerate(hits) if h)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(10, len(rel[q["id"]]))))
        ndcg.append(dcg / idcg)
        mrr.append(next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0))
        acc1.append(float(hits[0]))
    res = {"name": a.name, "model": a.model, "n_queries": len(qs), "n_corpus": len(docs),
           "ndcg@10": 100 * float(np.mean(ndcg)), "mrr@10": 100 * float(np.mean(mrr)), "acc@1": 100 * float(np.mean(acc1))}
    Path(a.out, f"csn_{a.name}.json").write_text(json.dumps(res, indent=1))
    log(f"csn {a.name}: nDCG@10 {res['ndcg@10']:.2f} MRR@10 {res['mrr@10']:.2f} Acc@1 {res['acc@1']:.2f} (n={len(qs)})")


if __name__ == "__main__":
    main()
