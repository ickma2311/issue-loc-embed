"""Split Python source into retrieval units: top-level functions, classes and methods.

Unit ids follow the LocAgent convention used by the SWE-Bench-Lite gold labels:
``path.py:func``, ``path.py:Class``, ``path.py:Class.method``, ``path.py:Outer.Inner.method``.
Nested functions are part of their enclosing function. Definitions inside module-level
``if``/``try``/``with`` blocks are treated as top level.
"""
from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_BLOCKS = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)
if hasattr(ast, "TryStar"):
    _BLOCKS = _BLOCKS + (ast.TryStar,)


@dataclass(frozen=True)
class Unit:
    uid: str          # "path.py:Class.method"
    kind: str         # "function" | "class" | "method"
    start: int        # first line (1-based, includes decorators)
    end: int          # last line (inclusive)
    indent: int       # column of the def/class keyword
    parent: str | None  # uid of the enclosing class, if any


def _span(node) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno


def parse_units(src: str, path: str) -> list[Unit] | None:
    """Return the units of one file, or None if it does not parse."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    out: list[Unit] = []

    def visit(body, prefix: str, parent: str | None, in_class: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s, e = _span(node)
                uid = f"{path}:{prefix}{node.name}"
                out.append(Unit(uid, "method" if in_class else "function", s, e, node.col_offset, parent))
            elif isinstance(node, ast.ClassDef):
                s, e = _span(node)
                uid = f"{path}:{prefix}{node.name}"
                out.append(Unit(uid, "class", s, e, node.col_offset, parent))
                visit(node.body, f"{prefix}{node.name}.", uid, True)
            elif isinstance(node, _BLOCKS) and not in_class:
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, []) or [], prefix, parent, in_class)
                for h in getattr(node, "handlers", []) or []:
                    visit(h.body, prefix, parent, in_class)

    visit(tree.body, "", None, False)
    return out


def unit_text(lines: list[str], u: Unit) -> str:
    """Source of a unit; ``lines`` from ``src.splitlines(keepends=True)``."""
    return "".join(lines[u.start - 1:u.end])
