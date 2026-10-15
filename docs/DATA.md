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
