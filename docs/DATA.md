# Training data

We do not redistribute code or issue text. GitHub issue text is user content that is not covered by the
repositories' licences, and MIT/BSD/Apache code would have to carry each project's copyright notice.

Instead we publish **pointers** (`data/train_pointers.parquet`, written by `scripts/export_pointers.py`): for every training task the SWE-rebench
instance id, repository, base commit and the modified function/method ids. Which pairs pass the consistency
filter is not included; re-running the mining step reproduces it — 43,068 kept pairs with vLLM on L4, 43,074
with the Hugging Face encoder on T4, probably from numerical differences near the cut-off (the original list was
not kept, so this was not checked pair by pair).

The pointer file is adapted from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2` (Nebius, CC-BY-4.0,
https://creativecommons.org/licenses/by/4.0/; changes: row selection and derived function ids) and is
distributed under CC-BY-4.0. `build_data.py` and `issueloc/paperfmt.py` rebuild the full training set from the public
sources (SWE-rebench / SWE-rebench-V2, CC-BY-4.0, and the repositories at their base commits).

Selection rules (identical to our run):

- tasks from `nebius/SWE-rebench` and `nebius/SWE-rebench-V2` (Python), real issue text only
  (never the LLM-written statements of `SWE-rebench-V2-PRs`);
- repository licence (per-row field) matching `issueloc/tasks.py::PERMISSIVE`: MIT, MIT-0, MIT-CMU, Apache-2.0,
  BSD variants, 0BSD, ISC, Unlicense, CC0-1.0, ZPL-2.1, PSF-2.0, BSL-1.0 (Boost), NCSA, PostgreSQL, AFL-3.0,
  HPND, zlib; anything else (including `custom-check-github` and dual-licence strings) is dropped;
- every repository of SWE-bench (test/dev), SWE-bench Lite and LocBench removed by name, plus 13 repositories
  that share a project name with an evaluation repository (likely renamed copies). A later GitHub check
  (`scripts/checks/fork_check.py`) found one renamed copy that the name rules missed: `Qiskit/qiskit-terra`
  (81 tasks) is the former name of LocBench's `Qiskit/qiskit`; `aeon-toolkit/aeon` is a hard fork of LocBench's
  `sktime`. LocBench headline numbers exclude those issues;
- at most 100 tasks per repository; after conversion 18,784 tasks from 3,378 repositories (3,379 names before case folding) with 82,613 positive
  function ids (`data/train_pointers.parquet`).

Dataset revisions read by the build (the auto-converted parquet branch; no later commit exists): see
`scripts/checks/hf_revisions.json`. Re-running the task selection on those revisions gives the same 20,574
selected tasks.

## v0.2: Python + 8 other languages

The v0.2 model was trained on the v0.1 Python tasks above plus tasks in Go, TypeScript, JavaScript, Rust, Java,
PHP, C and C++.

**Pointers.** `data/train_pointers_v02.parquet` (written by `scripts/export_pointers.py`): 33,338 tasks (python
18,784, go 4,802, ts 3,092, js 2,337, rust 2,197, java 1,212, php 820, c 56, cpp 38) with instance id, repository,
base commit, language, positive function ids, and `positives_kept` (whether each positive passed the consistency
filter in our run: 66,383 kept). `data/tasks_ml_v02.json` lists the 16,661 selected non-Python instance ids before
function extraction.

**Source and attribution.** The 18,784 Python rows of the v0.2 pointer file are the v0.1 pointers (sources and
revisions as above). The 14,554 rows in the other languages come from `nebius/SWE-rebench-V2` at revision
`10483de0f50fe5da545942705a76c6150171af7f` (Nebius, CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/;
"SWE-rebench V2", arXiv 2602.23866). Changes: row selection (rules below) and derived function ids. The v0.2
pointer file is distributed under CC-BY-4.0. Code and issue text are not redistributed; each instance remains
subject to its source repository's licence. Task and positive counts are before the consistency filter (142,282
positive function ids; mining scored 141,870 pairs, the other 412 positives belong to 76 tasks whose candidate
sample had no negative).

**Selection** (`scripts/select_ml.py`, the run that produced `data/tasks_ml_v02.json`):

- languages go, rust, java, php, c, cpp, js, ts; real issue text (`problem_statement`) of at least 20 characters;
- repository licence matching the v0.1 `PERMISSIVE` rule;
- the patch changes at least one non-test source file of the task's language (test files by the CoRNStack rule);
- held out: every repository of SWE-bench (test/dev), SWE-bench Lite, LocBench V0.1/V0.2/V1, SWE-bench
  Multilingual and `mteb/SWEbenchMultilingualRR` (270 repositories), with their current GitHub names; dropped: a
  training repository whose name, project name (part after "/"), current GitHub name, fork parent or fork source
  matches one of them (31 repositories; two only by the current name: `vuejs/vue-next` → `vuejs/core`,
  `vuepress/vuepress-next` → `vuepress/core`);
- issue text not equal (normalised) to any evaluation issue;
- one task per (current GitHub repository name, PR number); at most 100 tasks per current repository (seed 0).

**Units and text format** (`issueloc/units_ml.py`; tree-sitter 0.25.2, tree-sitter-language-pack 1.22.1, both MIT,
installed with pip, not included here). Units are functions and methods; nested functions belong to the enclosing
unit. Text: `path/Container/name` + newline + `class Container:` + the full source lines (methods of Java / PHP
classes, C++ classes and structs, JS / TS classes), or `path/name` + newline + source (Go, Rust and C functions,
other JS / TS functions). This is the convention of the SweRank+ / CoRNStack corpus in `mteb/SWEbenchMultilingualRR`,
which we read from that corpus. Texts are cut at 50,000 characters.

**Labels** (`issueloc/patch.py::modified_units`, the v0.1 rule): the innermost unit that contains a removed or
changed line, or an insertion point whose first inserted line is indented deeper than the unit's first line.
At most 20 positives per task. We did not validate these labels systematically. Known limits, with their size in
the kept training data (counted after training; the scores use the benchmarks' own gold labels):

| Limit | Effect | Size in the kept data |
|---|---|---|
| `const f = ..., g = ...` (several declarators in one statement) | all get the statement's span; a change to `g` can label `f` | 1 of 8,809 kept JS / TS positives |
| insertion inside a Go function with no deeper indent | no positive for that insertion | 168 such hunks in 88 of 3,400 kept Go tasks (upper bound by a line pattern) |
| functions stored as object properties (`key: () => ...`) | not a unit | 1,263 changed lines of this form in 381 of 4,363 kept JS / TS tasks |
| Rust tests inside source files (`#[test]`, `#[cfg(test)]`) | test functions are units and can be positives | 74 of 3,397 kept Rust positives have test-like names; test context in 201 of 1,350 kept Rust tasks |

**Merge.** `merge_data.py` converts the v0.1 Python build with `issueloc.paperfmt` (unchanged) and joins it with
the multi-language build (`build_data_ml.py`) into one data set: 33,338 tasks, 3,440,484 pool texts. The reused
Python tasks were not re-checked under the v0.2 rules (current names, forks); for example `python-babel/babel`
and `damnever/fmt` share a project name with `babel/babel` and `fmtlib/fmt`, which are not in the three
benchmarks used here.

**Isolation: what we checked, and the limits.**

- Isolation is by repository, plus exact and near-duplicate issue checks, with the known exceptions below. There is
  no time split. The current-name and fork rules above apply only to the new-language tasks; the Python tasks keep
  the v0.1 rules (names and 13 project-name drops). Known exceptions in the training data: `Qiskit/qiskit-terra` (81
  tasks) and `aeon-toolkit/aeon` (15 tasks), left out of the LocBench 552 numbers; `matchms/matchms` (23 tasks;
  LocBench's `matchms/matchms-backup`, 1 issue inside the 552, is a copy); `diofant/diofant` (3 tasks; a SymPy fork);
  `prometheus/tsdb` and `nushell/engine-q` (next item).
- Code lineage: two training repositories are code predecessors of SWE-bench Multilingual repositories, which the
  name, rename and fork rules missed: `prometheus/tsdb` (now `prometheus-junkyard/tsdb`; its README says it moved
  into `prometheus/prometheus`) with 3 Go tasks (5 kept positives), and `nushell/engine-q` (archived; merged into
  `nushell/nushell`) with 1 Rust task. 11 of the 234 SWE-bench Multilingual issues come from `prometheus/prometheus`
  (7) and `nushell/nushell` (4). A one-off heuristic screen (`scripts/checks/lineage_screen.py`) of all 2,110 non-Python repositories with training tasks (for archived repositories and repositories described as moved or merged, the README was searched for an evaluation repository name; 202 READMEs, no API failure) found no other lineage. It cannot find a lineage that the repository does not state. An earlier, broader screen of all 2,180 selected repositories also matched `babel/babel-eslint` (moved into `babel/babel`; no training task, not in these benchmarks) and one false match, a repository name inside a longer name; the screen now matches whole repository names.
- Near-duplicate issues: word 5-shingle Jaccard and containment (containment = shared shingles / shingles of the
  evaluation query) of every training issue (the v0.1 build, including the 13 repositories dropped later, and the
  16,661 selected V2 tasks) against every query of SWE-bench Multilingual, LocBench and SWE-Bench-Lite. A pair is
  flagged when Jaccard ≥ 0.5 or containment ≥ 0.5 and both texts have at least 20 shingles. Two evaluation issues
  were flagged: LocBench `jax-ml__jax-22897` equals a task of `google/jax` (the former name of `jax-ml/jax`), one of
  the 13 repositories dropped before training; one Multilingual issue shares only a new-issue editor template
  comment with five training issues. No other pair is flagged; the next largest values (Jaccard / containment) are 0.08 / 0.12 on SWE-bench Multilingual, 0.32 / 0.44 on LocBench and 0.23 / 0.35 on SWE-Bench-Lite.
- Base-model contamination is unknown (CoRNStack pre-training data).
