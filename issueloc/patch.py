"""Map a unified diff onto the units of the pre-patch (base_commit) files.

A unit counts as modified when (a) a removed/changed old line falls inside it (innermost unit
wins), or (b) lines are inserted after old line p and the first non-blank inserted line is
indented deeper than the unit's def/class line while p lies inside the unit (innermost wins).
A change inside a class but outside its methods therefore marks the class itself (this also
covers adding a new method to a class). Module-level changes mark nothing. Following the
SWE-Bench-Lite gold labels released by LocAgent, ``Class.__init__`` changes are reported as ``Class``.
Blank-line-only changes are ignored.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .units import Unit

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(patch: str) -> dict[str, dict]:
    """{old_path: {"changed": set(old lines), "inserts": [(after_old_line, indent)]}}.
    Hunk line counts are tracked so body lines that start with '--- ' / '+++ ' are not headers."""
    files: dict[str, dict] = {}
    cur = None
    old_ln = 0
    old_left = new_left = 0   # lines remaining in the current hunk
    pending_insert = None
    for raw in patch.splitlines():
        in_hunk = old_left > 0 or new_left > 0
        if not in_hunk:
            if raw.startswith("diff --git "):
                cur = None
                continue
            if raw.startswith("--- "):
                p = raw[4:].strip().split("\t")[0]
                cur = None if p == "/dev/null" else files.setdefault(p[2:] if p.startswith("a/") else p,
                                                                     {"changed": set(), "inserts": []})
                continue
            if raw.startswith("+++ "):
                continue
            m = _HUNK.match(raw)
            if m:
                old_ln = int(m.group(1))
                old_left = int(m.group(2)) if m.group(2) is not None else 1
                new_left = int(m.group(4)) if m.group(4) is not None else 1
                pending_insert = None
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        tag, body = (raw[0], raw[1:]) if raw else (" ", "")
        if tag == " ":
            old_ln += 1
            old_left -= 1
            new_left -= 1
            pending_insert = None
        elif tag == "-":
            if cur is not None and body.strip():
                cur["changed"].add(old_ln)
            old_ln += 1
            old_left -= 1
            pending_insert = None
        elif tag == "+":
            new_left -= 1
            if cur is not None and body.strip() and pending_insert is None:
                indent = len(body) - len(body.lstrip(" \t"))
                pending_insert = (old_ln - 1, indent)
                cur["inserts"].append(pending_insert)
        else:  # malformed line: leave the hunk
            old_left = new_left = 0
    return files


def modified_units(patch: str, units_by_path: dict[str, list[Unit]]) -> list[str]:
    """Unit ids touched by ``patch`` (sorted, unique). Files not in ``units_by_path`` are skipped."""
    hits: set[str] = set()
    for path, ch in parse_diff(patch).items():
        units = units_by_path.get(path)
        if not units:
            continue
        for ln in ch["changed"]:
            inner = [u for u in units if u.start <= ln <= u.end]
            if inner:
                hits.add(min(inner, key=lambda u: u.end - u.start).uid)
        for after, indent in ch["inserts"]:
            inner = [u for u in units if u.start <= after <= u.end and indent > u.indent]
            if inner:
                hits.add(min(inner, key=lambda u: u.end - u.start).uid)
    # LocAgent / SWE-Bench-Lite gold convention: a change to ``Class.__init__`` is labelled ``Class``
    hits = {h[: -len(".__init__")] if h.endswith(".__init__") else h for h in hits}
    return sorted(hits)


def touched_py_files(patch: str) -> list[str]:
    return [p for p in parse_diff(patch) if p.endswith(".py")]


def related(uid_a: str, uid_b: str) -> bool:
    """True when one unit encloses the other (same file, qualname prefix)."""
    if uid_a == uid_b:
        return True
    fa, qa = uid_a.split(":", 1)
    fb, qb = uid_b.split(":", 1)
    return fa == fb and (qa.startswith(qb + ".") or qb.startswith(qa + "."))


def group_by_path(units: list[Unit]) -> dict[str, list[Unit]]:
    d = defaultdict(list)
    for u in units:
        d[u.uid.split(":", 1)[0]].append(u)
    return d
