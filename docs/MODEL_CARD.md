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

# issue-loc-embed-small (v0.2)

A 137M bi-encoder that ranks the functions of a repository by how likely they need to change to resolve a GitHub
issue. It is `nomic-ai/CodeRankEmbed` (MIT) fine-tuned with an independent re-implementation of the SweRank data
recipe ([arXiv 2505.07849](https://arxiv.org/abs/2505.07849)) on permissively licensed repositories only. No SweRank
code or weights were used for training, filtering or mining. Licence: Apache-2.0.

**v0.2** (this revision, tag `v0.2`) was trained on Python and on Go, TypeScript, JavaScript, Rust, Java, PHP, C and
C++ issues. **v0.1** (Python only, the first release) stays available under the tag `v0.1`.

| Version | Load with |
|---|---|
| v0.2 (default) | `SentenceTransformer("ickma2311/issue-loc-embed-small", revision="v0.2", trust_remote_code=True)` (or no `revision`) |
| v0.1 | `SentenceTransformer("ickma2311/issue-loc-embed-small", revision="v0.1", trust_remote_code=True)` |

Code, evaluation protocol and training-data pointers: https://github.com/ickma2311/issue-loc-embed.

## Results (measured by us)

Protocol: SweRank/CoRNStack function localization. Candidates are the functions and methods of non-test files at
the issue's base commit, each represented as a `path/Class/function` (or `path/function`) id line plus its source;
Acc@k = all gold functions in the top k. All three models were run with sentence-transformers 5.7.0 and
transformers 4.57.6, fp16 on Kaggle T4, queries 1,024 tokens with the query prefix, functions 512 tokens. v0.1 and
v0.2 ran in the same session; CodeRankEmbed results come from earlier runs on the same inputs and settings.

| Benchmark | Issues | Model | Acc@1 | Acc@5 | Acc@10 |
|---|---|---|---|---|---|
| SWE-bench Multilingual | 234 | nomic-ai/CodeRankEmbed | 8.55 | 25.64 | 35.47 |
| SWE-bench Multilingual | 234 | issue-loc-embed-small v0.1 | 12.39 | 33.76 | 42.31 |
| SWE-bench Multilingual | 234 | **issue-loc-embed-small v0.2** | 13.68 | 34.19 | 45.30 |
| SWE-Bench-Lite (Python) | 274 | nomic-ai/CodeRankEmbed | 24.09 | 52.19 | 60.22 |
| SWE-Bench-Lite (Python) | 274 | issue-loc-embed-small v0.1 | 31.02 | 59.49 | 72.99 |
| SWE-Bench-Lite (Python) | 274 | **issue-loc-embed-small v0.2** | 31.75 | 61.68 | 74.09 |
| LocBench, 552 issues (Python) | 552 | nomic-ai/CodeRankEmbed | 20.47 | 38.77 | 46.74 |
| LocBench, 552 issues (Python) | 552 | issue-loc-embed-small v0.1 | 26.09 | 46.20 | 56.70 |
| LocBench, 552 issues (Python) | 552 | **issue-loc-embed-small v0.2** | 25.72 | 47.83 | 57.61 |
| LocBench, all 560 (Python) | 560 | nomic-ai/CodeRankEmbed | 20.89 | 39.29 | 47.32 |
| LocBench, all 560 (Python) | 560 | issue-loc-embed-small v0.1 | 26.43 | 46.79 | 57.32 |
| LocBench, all 560 (Python) | 560 | **issue-loc-embed-small v0.2** | 26.25 | 48.57 | 58.21 |

Paired bootstrap over issues (10,000 resamples, 95% interval). These intervals cover the sampling of benchmark
issues only, not training-run variance (one training run).

| Comparison | Acc@1 | Acc@5 | Acc@10 |
|---|---|---|---|
| SWE-bench Multilingual, v0.2 − v0.1 | +1.28 (−1.28 to +3.85) | +0.43 (−2.99 to +3.85) | +2.99 (+0.43 to +5.98) |
| SWE-Bench-Lite, v0.2 − v0.1 | +0.73 (−1.82 to +3.65) | +2.19 (−1.09 to +5.47) | +1.09 (−0.73 to +2.92) |
| LocBench (552), v0.2 − v0.1 | −0.36 (−1.45 to +0.72) | +1.63 (+0.00 to +3.44) | +0.91 (−0.54 to +2.54) |
| SWE-bench Multilingual, v0.2 − CodeRankEmbed | +5.13 (+2.14 to +8.55) | +8.55 (+4.70 to +12.82) | +9.83 (+5.56 to +14.10) |
| SWE-Bench-Lite, v0.2 − CodeRankEmbed | +7.66 (+2.92 to +12.41) | +9.49 (+5.11 to +13.87) | +13.87 (+9.49 to +18.61) |
| LocBench (552), v0.2 − CodeRankEmbed | +5.25 (+3.08 to +7.43) | +9.06 (+6.16 to +11.96) | +10.87 (+8.15 to +13.77) |

How to read this:

- **SWE-bench Multilingual: v0.2 − v0.1 = +2.99 Acc@10 (9 issues gained, 2 lost; one training run).** The difference
  is small: an exact paired test gives p = 0.065, and resampling the 38 repositories instead of the issues gives an
  interval of 0.00 to +5.91.
- Most of the gain over CodeRankEmbed on SWE-bench Multilingual (+6.84 Acc@10) was already in v0.1, which was
  trained on Python only, before we looked at this benchmark.
- **Python:** Acc@10 rose on both Python sets (no Acc@10 loss beyond 1.5 points); LocBench Acc@1 is 0.36 points
  lower than v0.1.
- We cannot say which part of the change caused the differences to v0.1: v0.2 added training data in 8 languages
  **and** trained for 604 steps instead of 286 (one run each).
- **Published reference:** SweRank+ (arXiv 2512.20482, Table 2) reports Acc@10 44.02 for SweRankEmbed-Small
  (CC-BY-NC-4.0) on SWE-bench Multilingual. v0.2 scores 45.30 on our pipeline. We did not run SweRankEmbed-Small on
  this benchmark, so this is not a head-to-head comparison.
- **LocBench, 552 issues:** we leave out the 8 issues from `sktime/sktime` and `Qiskit/qiskit`, because the training
  data contains a fork (`aeon-toolkit/aeon`) and a renamed copy (`Qiskit/qiskit-terra`) of those repositories. The
  numbers on all 560 issues are in the table.

**SWE-bench Multilingual by language group** (Acc@k, v0.2 / v0.1 / CodeRankEmbed):

| Language group | Issues | Training tasks | Acc@1 | Acc@5 | Acc@10 |
|---|---|---|---|---|---|
| C/C++ | 27 | 94 | 7.4 / 11.1 / 11.1 | 29.6 / 29.6 / 22.2 | 44.4 / 40.7 / 37.0 |
| Go | 29 | 4,802 | 24.1 / 13.8 / 10.3 | 31.0 / 27.6 / 27.6 | 41.4 / 37.9 / 41.4 |
| Java | 41 | 1,212 | 14.6 / 17.1 / 12.2 | 41.5 / 46.3 / 41.5 | 56.1 / 51.2 / 41.5 |
| JS/TS | 27 | 5,429 | 18.5 / 11.1 / 7.4 | 33.3 / 18.5 / 22.2 | 40.7 / 33.3 / 29.6 |
| PHP | 40 | 820 | 17.5 / 15.0 / 10.0 | 42.5 / 45.0 / 27.5 | 52.5 / 50.0 / 42.5 |
| Ruby | 36 | 0 | 8.3 / 8.3 / 2.8 | 22.2 / 22.2 / 11.1 | 33.3 / 36.1 / 25.0 |
| Rust | 34 | 2,197 | 5.9 / 8.8 / 5.9 | 35.3 / 38.2 / 23.5 | 44.1 / 41.2 / 29.4 |

The groups are small (27–41 issues). At Acc@10 the differences to v0.1 are gains of 1–2 issues per group, and
Ruby (no training data) is 1 issue lower. v0.2 is lower than v0.1 at Acc@5 on Java, PHP and Rust and at Acc@1 on
C/C++, Java and Rust. We make no claim for Ruby or for languages that are not in the training data. For Go, the
Acc@10 difference to v0.1 depends on one issue whose gold function is wrong in the benchmark (next section), so we
make no Go claim.

**Checks after the evaluation (post hoc, Acc@10, v0.2 / v0.1 / CodeRankEmbed).** These do not replace the numbers
above.

- Without the 11 issues from `prometheus/prometheus` and `nushell/nushell` (see code lineage below): 44.39 / 41.70 /
  34.53 (223 issues).
- Without the 4 Go issues whose released gold text is another method with the same name in the same file
  (`caddyserver__caddy-4943`, `caddyserver__caddy-6115`, `prometheus__prometheus-14861`, `prometheus__prometheus-15142`):
  45.65 / 43.04 / 36.09 (230 issues).

## Evaluation hygiene

- **One training run, one seed, one model family.** The intervals above cover benchmark-issue sampling only.
- **SWE-bench Multilingual is benchmark-informed for v0.2.** Before v0.2 was trained we had looked at v0.1 and
  CodeRankEmbed results on it (overall, by language, and issue by issue for Go), and we took the text format of the
  non-Python training units (`path/Class/name` with `class Class:`, or `path/name`) from its candidate corpus. The
  8 new training languages are 8 of its 9 languages (all but Ruby). No v0.2 checkpoint was evaluated before the final run; the step count and
  all data rules were fixed before training, and the released checkpoint is the last training step.
- **Isolation is by repository, plus exact and near-duplicate issue checks, with known exceptions.** There is no
  time split. Rules for the Python tasks (from v0.1): repositories of SWE-bench, SWE-bench Lite and LocBench removed
  by name, and 13 training repositories that share a project name with one of them. Rules for the new-language
  tasks: repositories of SWE-bench, SWE-bench Lite, LocBench and SWE-bench Multilingual removed by name, project
  name, current GitHub name, fork parent and fork source. Known exceptions in the training data:
  `Qiskit/qiskit-terra` (81 tasks; the former name of LocBench's `Qiskit/qiskit`) and `aeon-toolkit/aeon` (15
  tasks; a fork of LocBench's `sktime`), both left out of the LocBench 552 numbers; `prometheus/tsdb` and
  `nushell/engine-q` (code lineage, below); `matchms/matchms` (23 tasks), of which LocBench's `matchms/matchms-backup`
  (1 issue, inside the 552) is a copy; `diofant/diofant` (3 tasks; a SymPy fork; its positive function ids do not
  match SWE-Bench-Lite's SymPy gold ids). Several training repositories share an ecosystem with evaluation
  repositories (for example `dask/distributed` and LocBench's `dask/dask`).
- **Near-duplicate issues.** Word 5-shingle Jaccard and containment (shared shingles / shingles of the evaluation
  query) of every training issue against every evaluation query; a pair is flagged at 0.5 or more when both texts
  have at least 20 shingles. Two evaluation issues were flagged: one LocBench issue equals an issue of `google/jax`
  (the former name of `jax-ml/jax`), which was removed before training, and one SWE-bench Multilingual issue shares
  only a new-issue editor template comment with five training issues.
- **Copied code.** Three evaluation gold functions have an identical copy in the v0.1 training candidate pool
  (SWE-Bench-Lite's Requests `stream_decode_response_unicode` and Django `Q._combine`, LocBench's Dask
  `_compute_multiplier`; vendored or ported copies). None is a training positive; whether one was a hard negative is
  unknown. Semantic near-duplicates were not checked.
- **Code lineage that the repository rules missed.** The training data contains 3 Go tasks from `prometheus/tsdb`
  (the storage layer that moved into `prometheus/prometheus`) and 1 Rust task from `nushell/engine-q` (archived;
  merged into `nushell/nushell`). 11 of the 234 SWE-bench Multilingual issues come from `prometheus/prometheus` (7)
  and `nushell/nushell` (4) (post-hoc check above). For Python, `aeon-toolkit/aeon` and `Qiskit/qiskit-terra` are excluded from the LocBench
  552 numbers as before. A one-off heuristic screen of all 2,110 non-Python repositories with training tasks (for archived repositories and repositories described as moved or merged, the README was searched for an evaluation repository name; 202 READMEs, no API failure) found no other lineage. It cannot find a lineage that the repository does not state.
- **Benchmark gold errors.** In 4 SWE-bench Multilingual Go issues the gold text is the body of another method with
  the same name in the same file. With the correct method bodies, v0.1 and CodeRankEmbed both gain the same 3 issues;
  v0.2 was not re-scored on the corrected texts.
- **Training labels for the new languages were not validated systematically.** Known extraction limits and their
  size in the training data are listed in the code repository (docs/DATA.md); the benchmark scores use the
  benchmarks' own gold labels.
- **Base-model contamination unknown.** We do not know whether CodeRankEmbed's pre-training data (CoRNStack)
  contains the evaluation repositories.
- **Precision and side effects.** All numbers are fp16 on GPU; the default float32 loading in Usage was not
  evaluated. General code search was not measured for v0.2; for v0.1, CodeSearchNet Python (MTEB) nDCG@10 was 1.1
  points lower than CodeRankEmbed (94.39 vs 95.53).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ickma2311/issue-loc-embed-small", trust_remote_code=True)
units = ["src/pkg/io.py/Reader/read\nclass Reader:    def read(self, path):\n        ...",
         "internal/store/cache.go/Get\nfunc (c *Cache) Get(key string) (Item, bool) {\n\t...\n}",
         "src/main/java/org/x/Parser.java/Parser/parse\nclass Parser:    public Node parse(String s) {\n        ...\n    }"]
model.max_seq_length = 1024   # queries, as evaluated
q = model.encode([issue_text], prompt="Represent this query for searching relevant code: ",
                 normalize_embeddings=True)
model.max_seq_length = 512    # code units, as evaluated
d = model.encode(units, normalize_embeddings=True)
scores = q @ d.T
```

Represent each function as an id line, then its full source lines. Methods of classes (Python, Java, PHP, C++
classes and structs, JavaScript / TypeScript classes) use `path/Class/method` and the prefix `class Class:` before
the source; Go, Rust and C functions and other JavaScript / TypeScript functions use `path/function` (Go methods
have no receiver type in the id). In the code repository, `issueloc/units.py` with `issueloc/paperfmt.py` (Python)
and `issueloc/units_ml.py` (tree-sitter, other languages) produce exactly the format used for training. The query prefix is required. The
default `max_seq_length` in the configuration (8192) is CodeRankEmbed's; the results above use 1,024 for queries
and 512 for units. Same architecture and tokenizer as CodeRankEmbed.

## Demo

<video controls width="100%" src="https://huggingface.co/ickma2311/issue-loc-embed-small/resolve/main/demo/issue-loc-embed-demo.mp4"></video>

A 3-minute walkthrough made for v0.1 (Python): its results on SWE-Bench-Lite and LocBench, the training recipe,
usage, and two real issues. The numbers in the video are v0.1's. English narration, with English and Chinese
captions burned in. Caption files:
[English](https://huggingface.co/ickma2311/issue-loc-embed-small/blob/main/demo/issue-loc-embed-demo.en.srt), [Chinese](https://huggingface.co/ickma2311/issue-loc-embed-small/blob/main/demo/issue-loc-embed-demo.zh.srt).

## Training

- Data: issue → modified-function pairs from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2` (real issue text
  only), restricted to permissively licensed repositories. 33,338 tasks: the 18,784 Python tasks of v0.1 and 14,554
  tasks in Go (4,802), TypeScript (3,092), JavaScript (2,337), Rust (2,197), Java (1,212), PHP (820), C (56) and
  C++ (38) from SWE-rebench-V2 revision `10483de0f50fe5da545942705a76c6150171af7f`. Functions of the new languages
  were extracted with tree-sitter. Not redistributed: pointers (`data/train_pointers_v02.parquet`) and build scripts
  are in the code repository.
- Task and function counts above are before filtering (142,282 positive function ids). Mining scored 141,870
  issue-function pairs: the other 412 positives belong to 76 tasks whose candidate sample had no negative.
- Recipe (as v0.1): consistency filtering with CodeRankEmbed (66,383 of 141,870 issue-function pairs kept), 15
  mined hard negatives per pair, InfoNCE with in-batch negatives (batch 64, lr 2e-5, temperature 0.05), two rounds
  (round 2 re-mines negatives with the round-1 model and retrains from CodeRankEmbed), 604 steps per round (one
  epoch is 1,037 steps), on Kaggle T4.
- Base revision: `nomic-ai/CodeRankEmbed` at `3c4b60807d71f79b43f3c4363786d9493691f8b1`. The evaluation pinned this
  revision; the training runs loaded the repository's main branch without recording the revision, and main has had
  no other commit since 2025-06-24, so they used the same revision.

## Limitations

- One training run; the differences to v0.1 are small and mix more data with more training steps.
- Evaluated on Python and the 9 SWE-bench Multilingual languages only (there, C++ has 1 issue and TypeScript 3, by
  the benchmark's language tags). Little training data for C and C++ (94 tasks) and none for Ruby.
- Most of the gain needs the id-line unit format (v0.1 on SWE-Bench-Lite: +12.8 Acc@10 with it, +5.1 with source
  only; not re-measured for v0.2).
- Training labels for the new languages have known extraction limits (docs/DATA.md in the code repository).

## Licence and attribution

- Weights: Apache-2.0. Fine-tuned from `nomic-ai/CodeRankEmbed` (MIT License; see NOTICE).
- Training data was built from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2` (revision
  `10483de0f50fe5da545942705a76c6150171af7f` for the v0.2 languages), licensed CC-BY-4.0 by Nebius (Badertdinov et
  al., arXiv 2505.20411; "SWE-rebench V2", arXiv 2602.23866; https://creativecommons.org/licenses/by/4.0/). Changes:
  row selection and derived function ids. Code and issue text are not redistributed; the code repository publishes
  CC-BY-4.0 pointer files adapted from them. Each instance remains subject to its source repository's licence.
- `modeling_hf_nomic_bert.py`, `configuration_hf_nomic_bert.py`, tokenizer and configuration files are copied
  unchanged from `nomic-ai/CodeRankEmbed`, whose model card declares the MIT License (that repository has no
  LICENSE file; see NOTICE). The modeling file keeps its original header (Tri Dao, derived from the
  flash-attention BERT implementation).
- SweRankEmbed-Small (CC-BY-NC-4.0) appears only as a published reference value.

## Citation

- Method: Reddy et al., *SweRank: Software Issue Localization with Code Ranking*, arXiv 2505.07849.
- SWE-bench Multilingual reference values: *SweRank+*, arXiv 2512.20482.
- Evaluation protocol and base model data: Suresh et al., *CoRNStack*, arXiv 2412.01007.
- Training data: Badertdinov et al., *SWE-rebench: An Automated Pipeline for Task Collection and
  Decontaminated Evaluation of Software Engineering Agents*, arXiv 2505.20411; *SWE-rebench V2:
  Language-Agnostic SWE Task Collection at Scale*, arXiv 2602.23866.
- Evaluation labels: Chen et al., *LocAgent: Graph-Guided LLM Agents for Code Localization*, arXiv 2503.09089
  (SWE-Bench-Lite function labels and Loc-Bench); SWE-bench Multilingual via `mteb/SWEbenchMultilingualRR`.
