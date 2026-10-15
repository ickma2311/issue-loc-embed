---
license: apache-2.0
base_model: nomic-ai/CodeRankEmbed
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - code-retrieval
  - issue-localization
  - swe-bench
  - sentence-transformers
language:
  - en
---

<img src="https://huggingface.co/ickma2311/issue-loc-embed-small/resolve/main/chao-lab-logo-light-128.png" alt="Chao Lab" height="40">

# issue-loc-embed-small

A 137M bi-encoder that ranks the functions of a Python repository by how likely they need to change to
resolve a GitHub issue. It is `nomic-ai/CodeRankEmbed` (MIT) fine-tuned with an independent re-implementation
of the SweRank data recipe ([arXiv 2505.07849](https://arxiv.org/abs/2505.07849)) on permissively licensed
repositories only. No SweRank code or weights were used for training, filtering or mining (SweRankEmbed-Small
was only evaluated as a reference). Licence: Apache-2.0.

Code, evaluation protocol and training-data pointers: https://github.com/ickma2311/issue-loc-embed.

## Results (measured by us)

Protocol for both benchmarks: SweRank/CoRNStack function localization. Candidates are the functions and methods
of non-test files at the issue's base commit, each represented as `path/Class/function` + source; gold = the
benchmark's `edit_functions` (class-level labels mapped to `Class.__init__`); Acc@k = all gold functions in the
top k. Dense models use 1,024-token queries and 512-token units.

How each row was run:

- **SWE-Bench-Lite.** This model (the released file) and CodeRankEmbed: loaded with sentence-transformers as in
  Usage below, but cast to fp16 (Usage runs in the default float32, which we did not evaluate); at Acc@5, @10 and
  @20 every issue has the same hit / miss as in our earlier vLLM fp16 run. Qwen3, the round-1 model and
  SweRankEmbed-Small: vLLM (Qwen3 bf16, the others fp16). LateOn-Code: PyLate with the model in float32 and
  embeddings stored in fp16 for MaxSim scoring.
- **LocBench.** All three models: sentence-transformers, fp16. Qwen3's Lite (vLLM bf16) and LocBench
  (sentence-transformers fp16) numbers were not cross-checked against each other.

**SWE-Bench-Lite** (274 issues):

| Model | Licence | Params | Acc@5 | Acc@10 | Acc@20 |
|---|---|---|---|---|---|
| nomic-ai/CodeRankEmbed | MIT | 137M | 52.19 | 60.22 | 71.53 |
| lightonai/LateOn-Code ¹ | Apache-2.0 | 149M | 53.65 | 60.95 | — |
| Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | 596M | 55.84 | 65.33 | 75.55 |
| round-1 model (not released) | — | 137M | 63.14 | 70.07 | 78.10 |
| **issue-loc-embed-small** | Apache-2.0 | 137M | 59.49 | 72.99 | 78.83 |
| Salesforce/SweRankEmbed-Small (reference) | CC-BY-NC-4.0 | 137M | 62.04 | 74.82 | 81.75 |

¹ LateOn-Code: 256-token queries (its default) and units truncated to 512 tokens (its default is 2,048), which may
understate it.

**LocBench** (Loc-Bench V1; a second benchmark, not used for tuning). Headline numbers are on 552 of the 560
issues: the 8 issues from `Qiskit/qiskit` and `sktime/sktime` were excluded before scoring because training
contains a renamed copy or a fork of those repositories (see Evaluation hygiene).

| Model | Acc@5 | Acc@10 | Acc@15 | File Acc@5 | File Acc@10 | Acc@10, all 560 |
|---|---|---|---|---|---|---|
| nomic-ai/CodeRankEmbed | 38.77 | 46.74 | 49.82 | 73.55 | 79.17 | 47.32 |
| Qwen/Qwen3-Embedding-0.6B | 40.04 | 48.37 | 53.99 | 74.82 | 82.61 | 48.93 |
| **issue-loc-embed-small** | 46.20 | 56.70 | 62.14 | 78.44 | 83.70 | 57.32 |

Paper-reported on LocBench (all 560, SweRank Table 2, function Acc@10 / Acc@15): CodeRankEmbed 47.86 / 50.89,
SweRankEmbed-Small 58.57 / 63.39. We did not run SweRankEmbed-Small on LocBench.

Paired bootstrap over issues (10,000 resamples, 95% CI), this model minus the other (the SweRankEmbed-Small
comparison pairs our sentence-transformers outputs with SweRankEmbed-Small's vLLM run):

| Comparison | Acc@5 | Acc@10 | Acc@20 (Lite) / Acc@15 (LocBench) |
|---|---|---|---|
| Lite, vs CodeRankEmbed | +7.3 (2.9 to 11.7) | +12.8 (8.4 to 17.2) | +7.3 (4.0 to 11.0) |
| Lite, vs SweRankEmbed-Small | −2.6 (−7.3 to +2.2) | −1.8 (−6.2 to +2.2) | −2.9 (−6.2 to 0.0) |
| LocBench (552), vs CodeRankEmbed | +7.4 (4.5 to 10.3) | +10.0 (7.1 to 13.0) | +12.3 (9.4 to 15.2) |
| LocBench (552), vs Qwen3-Embedding-0.6B | +6.2 (2.7 to 9.4) | +8.3 (4.9 to 11.6) | +8.2 (4.7 to 11.4) |

Against SweRankEmbed-Small on SWE-Bench-Lite the Acc@10 difference is −1.82 points (paired 95% CI −6.20 to
+2.19); this experiment does not establish that the two models are equivalent. These intervals cover the
sampling of benchmark issues only (one training run, see Evaluation hygiene). In our single run, round 2 scores
higher than round 1 at Acc@10 (+2.9, CI 0.7 to 5.5) and lower at Acc@5 (−3.6, CI −6.6 to −1.1); one run does not
show whether the second mining round helps. Ablation (one run per arm): round 1 without the consistency filter
(same steps, same number of training pairs) reaches 56.57 Acc@10 vs 70.07 with it. Qwen3 instruction:
`Instruct: Given a GitHub issue, retrieve the code that needs to be changed to resolve it\nQuery:`.

| Robustness / side effects (SWE-Bench-Lite unless noted) ² | CodeRankEmbed | this model |
|---|---|---|
| Acc@10, units as source code only | 50.73 | 55.84 |
| Acc@10, test files included | 42.70 | 62.04 |
| CodeSearchNet Python (MTEB) nDCG@10 | 95.53 | 94.39 |

² From the retained summary outputs of those runs (vLLM); per-query results were not kept.

## Evaluation hygiene

- **One training run, one model family.** The model was trained once (seed 0; a second run with the same seed
  gave the same evaluation metrics). The confidence intervals above come from resampling the benchmark issues
  only; they say nothing about run-to-run training variance or about other base models.
- **No separate validation set.** The evaluation protocol (candidate units, unit text format, gold mapping) was
  fixed by reproducing the SweRank paper's numbers for CodeRankEmbed and SweRankEmbed-Small on SWE-Bench-Lite,
  that is, after looking at Lite results of those reference models (not of our models). The training text
  format was then aligned to the same SweRank / CoRNStack format. Hyper-parameters, step count and the released
  checkpoint (the final step of round 2) were fixed before any of our models was evaluated; no checkpoint or
  seed was selected on a benchmark. LocBench was not used for tuning; its comparison, test and exclusions were
  fixed before scoring.
- **No temporal holdout.** Training and evaluation repositories are disjoint by name, but the issue dates
  overlap: training issues 2014-04-30 to 2025-07-22, SWE-Bench-Lite 2012-09-20 to 2023-07-17, LocBench
  2015-11-07 to 2025-02-07. Several training repositories share an organisation or ecosystem with evaluation
  repositories (for example `dask/distributed` and LocBench's `dask/dask`).
- **What we checked, and the limits.**
  - Repository names (case-insensitive) against all SWE-bench, SWE-bench Lite and LocBench repositories, plus 13
    training repositories dropped for sharing a project name with one. A GitHub API check of fork parents and
    renames of all 3,378 training repositories found that `Qiskit/qiskit-terra` (81 training tasks) is the
    former name of LocBench's `Qiskit/qiskit`. This check cannot find hard (detached) forks, and one evaluation
    repository no longer exists on GitHub.
  - Hard forks found by name: `aeon-toolkit/aeon` (a fork of LocBench's `sktime`, excluded above) and
    `diofant/diofant` (a SymPy fork, 3 training tasks; mapping their positive function ids from `diofant/` to
    `sympy/` gives no match with SWE-Bench-Lite's SymPy gold ids). After scoring we also noticed that LocBench's
    `matchms/matchms-backup` (1 issue) is a copy of the training repository `matchms/matchms`; the headline keeps
    the 552 issues fixed before scoring, and without that issue (551) the difference to CodeRankEmbed is +9.98
    (7.08 to 13.07).
  - Instance ids, base commits and gold function ids (`path:qualname`) do not intersect with training on either
    benchmark. Issue text was compared lexically only (8-word shingles; the largest share of an evaluation issue
    found in one training issue is 0.29 on Lite and 0.39 on LocBench); one training issue links to a LocBench
    pull request (`numpy/numpy#10615`, from an unrelated repository).
  - Function bodies (whitespace-normalised, at least 200 characters): three evaluation gold functions have an
    identical copy in the training candidate pool — SWE-Bench-Lite's Requests `stream_decode_response_unicode`
    (vendored Requests copies), Django's `Q._combine` (a port in another project) and LocBench's Dask
    `_compute_multiplier` (a vendored Dask copy). None was ever a training positive; whether any was picked as a
    hard negative in the actual run is unknown, because the mined training records were not kept. Semantic
    near-duplicates were not checked.
- **Base-model contamination unknown.** We do not know whether CodeRankEmbed's pre-training data (CoRNStack)
  contains the evaluation repositories.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ickma2311/issue-loc-embed-small", trust_remote_code=True)
units = ["src/pkg/io.py/Reader/read\nclass Reader:    def read(self, path):\n        ...",
         "src/pkg/util.py/normalize\ndef normalize(x):\n    ..."]
model.max_seq_length = 1024   # queries, as evaluated
q = model.encode([issue_text], prompt="Represent this query for searching relevant code: ",
                 normalize_embeddings=True)
model.max_seq_length = 512    # code units, as evaluated
d = model.encode(units, normalize_embeddings=True)
scores = q @ d.T
```

Represent each function as its `path/Class/function` id on the first line, then its source (methods prefixed
with `class <Class>:`). The query prefix is required. The default `max_seq_length` in the configuration (8192)
is CodeRankEmbed's; the results above use 1,024 for queries and 512 for units. Same architecture and tokenizer
as CodeRankEmbed. Evaluated with sentence-transformers 5.7.0 and transformers 4.57.6, cast to fp16 on a GPU
(`model.half()`); the model also
loads with transformers 5.16.1.

## Demo

<video controls width="100%" src="https://huggingface.co/ickma2311/issue-loc-embed-small/resolve/main/demo/issue-loc-embed-demo.mp4"></video>

A 3-minute walkthrough: how much the model improves on SWE-Bench-Lite and LocBench, why the training recipe works,
how to use the model, and two real issues (one where it finds the changed function and one where it misses).
English narration, with English and Chinese captions burned in. Caption files:
[English](https://huggingface.co/ickma2311/issue-loc-embed-small/blob/main/demo/issue-loc-embed-demo.en.srt), [Chinese](https://huggingface.co/ickma2311/issue-loc-embed-small/blob/main/demo/issue-loc-embed-demo.zh.srt).

## Training

- Data: issue → modified-function pairs from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2` (Python, real
  issue text only), restricted to permissively licensed repositories (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0
  and others; full list in the code repository's docs/DATA.md). All SWE-bench, SWE-bench Lite and LocBench
  repositories were removed by name (see Evaluation hygiene for copies found later). 18,784 tasks. Not
  redistributed: pointers (`data/train_pointers.parquet`) and a build script are in the code repository.
- Recipe: consistency filtering with CodeRankEmbed (keep a pair if the positive ranks in the top K of a
  file-level sample of the repository, about 300 functions per snapshot, K = 20 scaled to the sample size), 15
  mined hard negatives per (issue, positive) pair from the same sample, InfoNCE with in-batch negatives (batch
  64, lr 2e-5, temperature 0.05), two rounds (round 2 re-mines negatives with the round-1 model and retrains from
  CodeRankEmbed), 286 steps per round on one L4 GPU.

## Limitations

- Python only.
- Most of the gain needs the `path/Class/function` unit format (+12.8 Acc@10 on SWE-Bench-Lite vs +5.1 with
  source only).
- Slightly worse than CodeRankEmbed on general docstring-to-code search.
- About 22% of the positive labels (before filtering) are class-level labels mapped to `Class.__init__` (noisy).
- Short training (about 43% of an epoch per round), one training run.

## Licence and attribution

- Weights: Apache-2.0. Fine-tuned from `nomic-ai/CodeRankEmbed` (MIT License).
- Training data was built from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2`, licensed CC-BY-4.0 by Nebius
  (Badertdinov et al., arXiv 2505.20411; "SWE-rebench V2", arXiv 2602.23866;
  https://creativecommons.org/licenses/by/4.0/). Code and issue text are not redistributed; the code repository
  publishes a CC-BY-4.0 pointer file adapted from them (row selection and derived function ids). Each instance
  remains subject to its source repository's licence.
- `modeling_hf_nomic_bert.py`, `configuration_hf_nomic_bert.py`, tokenizer and configuration files are
  copied unchanged from `nomic-ai/CodeRankEmbed`, whose model card declares the MIT License (that repository
  has no LICENSE file; see NOTICE). The modeling file keeps its original header (Tri Dao, derived from the
  flash-attention BERT implementation).
- SweRankEmbed-Small (CC-BY-NC-4.0) appears in the tables only as an evaluation reference.

## Citation

- Method: Reddy et al., *SweRank: Software Issue Localization with Code Ranking*, arXiv 2505.07849.
- Evaluation protocol and base model data: Suresh et al., *CoRNStack*, arXiv 2412.01007.
- Training data: Badertdinov et al., *SWE-rebench: An Automated Pipeline for Task Collection and
  Decontaminated Evaluation of Software Engineering Agents*, arXiv 2505.20411; *SWE-rebench V2:
  Language-Agnostic SWE Task Collection at Scale*, arXiv 2602.23866.
- Evaluation labels: Chen et al., *LocAgent: Graph-Guided LLM Agents for Code Localization*, arXiv 2503.09089
  (SWE-Bench-Lite function labels and Loc-Bench).
