"""Select the non-Python training tasks of v0.2 from SWE-rebench V2 and write the leakage check (the selection
that produced data/tasks_ml_v02.json; known gaps are listed in docs/DATA.md).
Usage: python -I scripts/select_ml.py <v2.parquet> <repository root> <out dir>
Writes <out>/leakage.json (evaluation repositories, dropped training repositories and why) and
<out>/tasks_ml.json (selected instance ids per language, after the cap of 100 per repository)."""
import collections, json, random, re, subprocess, sys, urllib.request, io
import pyarrow.parquet as pq

V2_REV = "10483de0f50fe5da545942705a76c6150171af7f"
LANGS = {"go", "rust", "java", "php", "c", "cpp", "js", "ts"}
JS = (".js", ".jsx", ".mjs", ".cjs")
EXT = {"go": (".go",), "rust": (".rs",), "java": (".java",), "php": (".php",), "c": (".c", ".h"),
       "cpp": (".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".h"), "js": JS, "ts": (".ts", ".tsx") + JS}
EVAL = [("princeton-nlp/SWE-bench", ["test", "dev"]), ("princeton-nlp/SWE-bench_Lite", ["test", "dev"]),
        ("czlll/Loc-Bench_V1", ["test"]), ("czlll/Loc-Bench_V0.2", ["test"]), ("czlll/Loc-Bench_V0.1", ["test"]),
        ("SWE-bench/SWE-bench_Multilingual", ["test"])]
HF = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/default/{}/0000.parquet"


def read(url, cols):
    return pq.read_table(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()), columns=cols).to_pylist()


def norm_issue(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def main(v2f, root, out):
    sys.path.insert(0, root)
    from issueloc.tasks import PERMISSIVE, s_, pr_key
    from issueloc.paperfmt import is_test_cornstack
    ev_repos, ev_issues = set(), set()
    for ds, splits in EVAL:
        for s in splits:
            rows = read(HF.format(ds, s), ["repo", "problem_statement"])
            ev_repos |= {r["repo"].lower() for r in rows}
            ev_issues |= {norm_issue(r["problem_statement"]) for r in rows}
    ml_meta = json.load(open(f"{root}/data/ml_meta.json"))
    ev_repos |= {r.lower() for r in ml_meta["repos"]}
    ev_names = {r.split("/")[1] for r in ev_repos}
    rows = pq.read_table(v2f, columns=["instance_id", "repo", "language", "license", "problem_statement", "patch",
                                       "base_commit"]).to_pylist()
    stats, kept, seen = collections.Counter(), [], set()
    drop_repo = collections.defaultdict(set)
    for r in rows:
        L = r["language"]
        if L not in LANGS:
            continue
        stats[f"{L}:all"] += 1
        repo = r["repo"]
        if repo.lower() in ev_repos:
            drop_repo[repo].add("eval_repo"); stats[f"{L}:drop_eval_repo"] += 1; continue
        if repo.split("/")[1].lower() in ev_names:
            drop_repo[repo].add("eval_project_name"); stats[f"{L}:drop_eval_name"] += 1; continue
        if not PERMISSIVE.match(s_(r["license"]).strip()):
            stats[f"{L}:drop_licence"] += 1; continue
        if len(s_(r["problem_statement"]).strip()) < 20:
            stats[f"{L}:drop_empty_issue"] += 1; continue
        if norm_issue(r["problem_statement"]) in ev_issues:
            stats[f"{L}:drop_eval_issue_text"] += 1; continue
        files = re.findall(r"^diff --git a/(\S+) b/", s_(r["patch"]), flags=re.M)
        if not [f for f in files if f.endswith(EXT[L]) and not is_test_cornstack(f, repo)]:
            stats[f"{L}:drop_no_src_change"] += 1; continue
        k = pr_key(repo, r["instance_id"])
        if k in seen:
            stats[f"{L}:drop_dup"] += 1; continue
        seen.add(k)
        kept.append(r)
    # renames, transfers and forks: GitHub state of every kept training repository
    # and of every evaluation repository; the current full name is the key for de-duplication and the cap
    def gh(repo):
        p = subprocess.run(["gh", "api", f"repos/{repo}", "--jq",
                            "[.full_name, .fork, (.parent.full_name // \"\"), (.source.full_name // \"\")] | @tsv"],
                           capture_output=True, text=True)
        if p.returncode != 0 or not p.stdout.strip():
            return None
        full, fork, parent, source = (p.stdout.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
        return {"full": full.lower(), "fork": fork == "true", "parent": parent.lower(), "source": source.lower()}
    ev_now = {r: gh(r) for r in sorted(ev_repos)}
    ev_all = ev_repos | {v["full"] for v in ev_now.values() if v}
    ev_names_all = ev_names | {x.split("/")[1] for x in ev_all}
    repos = sorted({r["repo"] for r in kept})
    info = {r: gh(r) for r in repos}
    renamed = {r: (v["full"] if v else "gh_error") for r, v in info.items() if not v or v["full"] != r.lower()}
    bad = set()
    for repo, v in info.items():
        if not v:
            continue
        for k in ("full", "parent", "source"):
            x = v[k]
            if x and (x in ev_all or x.split("/")[1] in ev_names_all):
                drop_repo[repo].add(f"{k}_is_eval:{x}"); bad.add(repo)
    kept = [r for r in kept if r["repo"] not in bad]
    cur = {r: (info[r]["full"] if info[r] else r.lower()) for r in repos}
    seen2, kept2 = set(), []
    for r in kept:  # same PR under two V2 names of one repository
        k = (cur[r["repo"]], pr_key(r["repo"], r["instance_id"])[1])
        if k in seen2:
            stats["drop_dup_current_name"] += 1; continue
        seen2.add(k); kept2.append(r)
    kept = kept2
    by_repo = collections.defaultdict(list)
    for r in kept:
        by_repo[cur[r["repo"]]].append(r)
    rng = random.Random(0)
    final = []
    for repo in sorted(by_repo):
        v = sorted(by_repo[repo], key=lambda r: r["instance_id"])
        if len(v) > 100:
            stats["capped"] += len(v) - 100
            v = sorted(rng.sample(v, 100), key=lambda r: r["instance_id"])
        final.extend(v)
    forks = {r: v["parent"] for r, v in info.items() if v and v["fork"]}
    per_lang = collections.Counter(r["language"] for r in final)
    json.dump({"v2_revision": V2_REV, "eval_sources": [d for d, _ in EVAL] + ["mteb/SWEbenchMultilingualRR (ml_meta.json)"],
               "n_eval_repos": len(ev_repos), "eval_repos": sorted(ev_repos), "n_eval_issue_texts": len(ev_issues),
               "dropped_training_repos": {r: sorted(w) for r, w in sorted(drop_repo.items())},
               "renamed_repos_checked": len(repos), "renamed": renamed, "gh_errors": sum(v == "gh_error" for v in renamed.values()),
               "forks": forks, "eval_repos_current": {k: (v["full"] if v else "gh_error") for k, v in ev_now.items()},
               "stats": dict(sorted(stats.items())), "final_tasks": len(final), "final_per_language": dict(per_lang.most_common()),
               "final_repos": len({r["repo"] for r in final}), "final_repos_current": len({cur[r["repo"]] for r in final})}, open(f"{out}/leakage.json", "w"), indent=1)
    json.dump({L: [r["instance_id"] for r in final if r["language"] == L] for L in sorted(per_lang)},
              open(f"{out}/tasks_ml.json", "w"))
    print(json.dumps({"final": len(final), "per_language": dict(per_lang.most_common()), "renamed": len(renamed),
                      "dropped_repos": len(drop_repo), "dropped_by_github_state": sorted(bad), "forks": len(forks)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
