"""Fetch repository snapshots at given commits (depth 1, large blobs filtered) and read .py files."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_ASKPASS="true")


def _git(cwd, *args, timeout=600, input_=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, env=ENV, capture_output=True, timeout=timeout,
                       input=input_)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace')[-300:]}")
    return r.stdout


def fetch(repo: str, shas: list[str], workdir: Path, timeout: int = 900) -> tuple[Path, list[str]]:
    """Shallow-fetch ``shas`` of github.com/<repo> into a fresh bare-ish repo. Returns (dir, fetched shas)."""
    d = workdir / repo.replace("/", "__")
    _PRESENT.pop(str(d), None)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    _git(d, "init", "-q")
    _git(d, "remote", "add", "origin", f"https://github.com/{repo}.git")
    for i in range(0, len(shas), 40):
        chunk = shas[i:i + 40]
        r = subprocess.run(["git", "-c", "protocol.version=2", "fetch", "-q", "--depth", "1",
                            "--filter=blob:limit=1m", "--no-tags", "origin", *chunk],
                           cwd=d, env=ENV, capture_output=True, timeout=timeout)
        if r.returncode != 0:  # one bad sha fails the whole command: retry one by one
            for s in chunk:
                subprocess.run(["git", "-c", "protocol.version=2", "fetch", "-q", "--depth", "1",
                                "--filter=blob:limit=1m", "--no-tags", "origin", s],
                               cwd=d, env=ENV, capture_output=True, timeout=timeout)
    # stop treating origin as a promisor so missing (>1 MB) blobs are reported, never lazily fetched
    _git(d, "config", "--unset-all", "extensions.partialclone", check=False)
    _git(d, "config", "--unset-all", "remote.origin.promisor", check=False)
    got = [s for s in shas
           if subprocess.run(["git", "cat-file", "-e", f"{s}^{{commit}}"], cwd=d, env=ENV,
                             capture_output=True).returncode == 0]
    return d, got


def list_py(d: Path, sha: str) -> list[tuple[str, str]]:
    """[(path, blob_sha)] for every .py blob in the tree of ``sha``."""
    out = _git(d, "ls-tree", "-r", "-z", sha)
    res = []
    for ent in out.split(b"\0"):
        if not ent:
            continue
        meta, path = ent.split(b"\t", 1)
        mode, typ, obj = meta.split(b" ")
        p = path.decode("utf-8", errors="replace")
        if typ == b"blob" and p.endswith(".py") and mode != b"120000":
            res.append((p, obj.decode()))
    return res


_PRESENT: dict[str, set[str]] = {}


def present_objects(d: Path) -> set[str]:
    """Objects stored locally. Blobs left out by --filter=blob:limit are absent; asking cat-file for
    them makes git try (and, with lazy fetch disabled, fatally fail) a promisor fetch."""
    key = str(d)
    if key not in _PRESENT:
        out = _git(d, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)", timeout=1200)
        _PRESENT[key] = set(out.decode().split())
    return _PRESENT[key]


def read_blobs(d: Path, objs: list[str]) -> dict[str, str]:
    """{blob_sha: text}; missing (filtered) or binary blobs are skipped."""
    have = present_objects(d)
    objs = [o for o in objs if o in have]
    if not objs:
        return {}
    out = _git(d, "cat-file", "--batch", input_=("\n".join(objs) + "\n").encode(), timeout=1200)
    res, i = {}, 0
    while i < len(out):
        nl = out.index(b"\n", i)
        header = out[i:nl].split()
        i = nl + 1
        if len(header) < 3 or header[1] == b"missing":
            continue
        size = int(header[2])
        data = out[i:i + size]
        i += size + 1
        if b"\0" in data[:8000]:
            continue
        res[header[0].decode()] = data.decode("utf-8", errors="replace")
    return res
