"""One-off heuristic lineage screen (read-only, gh api): for every non-Python repository with at least one v0.2 training
task (merged train_tasks_paper.parquet), read its GitHub state; for known pairs, and for archived repositories or
repositories whose description says moved / merged / deprecated, read the README and look for an evaluation
repository name (whole-name match). Records every API failure. Writes lineage_screen.json.
Usage: python -I scripts/checks/lineage_screen.py <train_tasks_paper.parquet> <leakage.json from scripts/select_ml.py>"""
import json, re, subprocess, sys
# known code lineages that the name / rename / fork rules miss, and the moved / merged wording
KNOWN_LINEAGE = {"prometheus/tsdb": "prometheus/prometheus", "prometheus-junkyard/tsdb": "prometheus/prometheus",
                 "nushell/engine-q": "nushell/nushell"}
MOVED = re.compile(r"\b(moved|merged|migrated|deprecated|archived|superseded|now (lives|part of)|has been (moved|merged))\b", re.I)
import pyarrow.parquet as pq

tasks = pq.read_table(sys.argv[1], columns=["repo", "language"]).to_pylist()
repos = sorted({t["repo"] for t in tasks if t["language"] != "python"})
lk = json.load(open(sys.argv[2]))
ev = set(lk["eval_repos"]) | {v for v in lk["eval_repos_current"].values() if v != "gh_error"}
pat = {e: re.compile(r"(?<![\w.-])" + re.escape(e) + r"(?![\w-])") for e in ev}


def gh(args):
    p = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    return (p.stdout if p.returncode == 0 else None)


res = {"scope": "non-Python repositories with at least one v0.2 training task", "repos_in_scope": len(repos),
       "meta_ok": 0, "meta_failed": [], "screened_readme": [], "readme_failed": [], "archived": [], "lineage": {},
       "moved_without_eval_mention": []}
for repo in repos:
    m = gh([f"repos/{repo}", "--jq", "[.full_name, .archived, (.description // \"\")] | @tsv"])
    if m is None or len(m.rstrip("\n").split("\t")) < 3:
        res["meta_failed"].append(repo); continue
    res["meta_ok"] += 1
    _, archived, desc = m.rstrip("\n").split("\t")[:3]
    archived = archived == "true"
    if archived:
        res["archived"].append(repo)
    known = repo.lower() in KNOWN_LINEAGE
    if not (known or archived or MOVED.search(desc)):
        continue
    rd = gh([f"repos/{repo}/readme", "-H", "Accept: application/vnd.github.raw"])
    if rd is None:
        res["readme_failed"].append(repo); rd = ""
    res["screened_readme"].append(repo)
    text = desc + "\n" + rd
    low = text.lower()
    hit = sorted(e for e, rx in pat.items() if rx.search(low))
    if known:
        res["lineage"][repo] = KNOWN_LINEAGE[repo.lower()]
    elif hit and MOVED.search(text):
        res["lineage"][repo] = hit[0]
    elif MOVED.search(text):
        res["moved_without_eval_mention"].append(repo)
json.dump(res, open("lineage_screen.json", "w"), indent=1)
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}))
