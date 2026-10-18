"""Split non-Python source into retrieval units with tree-sitter (training data of v0.2).

Units are functions and methods (nested functions belong to the enclosing unit). Unit ids: ``path:Container.name``
or ``path:name``; a repeated id in one file gets ``#2``, ``#3``, ... Texts follow the SweRank+ / CoRNStack corpus
convention: ``path/Container/name\\nclass Container:<source>`` or ``path/name\\n<source>``, source = the full lines
of the unit. Parser: tree-sitter 0.25.2 + tree-sitter-language-pack 1.22.1.
"""
from __future__ import annotations

from .units import Unit

JS = (".js", ".jsx", ".mjs", ".cjs")
EXT = {"go": (".go",), "rust": (".rs",), "java": (".java",), "php": (".php",), "c": (".c", ".h"),
       "cpp": (".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".h"), "js": JS, "ts": (".ts", ".tsx") + JS}
CHAR_CAP = 50000

FUNC = {
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "java": {"method_declaration", "constructor_declaration", "compact_constructor_declaration"},
    "php": {"method_declaration", "function_definition"},
    "javascript": {"function_declaration", "generator_function_declaration", "method_definition"},
}
FUNC["typescript"] = FUNC["tsx"] = FUNC["javascript"]
CONTAINER = {
    "cpp": {"class_specifier", "struct_specifier"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
    "php": {"class_declaration", "trait_declaration", "interface_declaration", "enum_declaration"},
    "javascript": {"class_declaration", "class"},
}
CONTAINER["typescript"] = CONTAINER["tsx"] = CONTAINER["javascript"] | {"abstract_class_declaration"}
VALUE_FUNC = {"arrow_function", "function_expression", "function", "generator_function"}

_PARSERS: dict = {}


def grammar(lang: str, path: str) -> str:
    """tree-sitter grammar for a file of a task language."""
    if lang in ("js", "ts"):
        if path.endswith(".tsx"):
            return "tsx"
        return "typescript" if path.endswith(".ts") else "javascript"
    return lang


def parser(g: str):
    if g not in _PARSERS:
        from tree_sitter_language_pack import get_parser
        _PARSERS[g] = get_parser(g)
    return _PARSERS[g]


def _ident(node, src: bytes):
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _c_name(decl, src):
    """Function name from a C / C++ declarator chain (last identifier part)."""
    d = decl
    for _ in range(20):
        if d is None:
            return None
        if d.type in ("identifier", "field_identifier", "operator_name", "destructor_name"):
            return _ident(d, src)
        if d.type == "qualified_identifier":
            n = d.child_by_field_name("name")
            return _c_name(n, src) if n is not None else None
        if d.type == "template_function":
            d = d.child_by_field_name("name"); continue
        nxt = d.child_by_field_name("declarator")
        if nxt is None:  # e.g. reference_declarator ("T& f()") has no field name
            nxt = next((c for c in d.named_children if c.type.endswith("declarator") or c.type in
                        ("identifier", "field_identifier", "qualified_identifier", "operator_name")), None)
        d = nxt
    return None


def _name(node, g: str, src: bytes):
    if g in ("c", "cpp"):
        return _c_name(node.child_by_field_name("declarator"), src)
    n = node.child_by_field_name("name")
    return _ident(n, src) if n is not None else None


def _container_name(node, src):
    n = node.child_by_field_name("name")
    return _ident(n, src) if n is not None else None


def parse_units(src_text: str, path: str, lang: str):
    """[(Unit, text)] for one file, or None when the file does not parse. ``path`` is the repository path."""
    g = grammar(lang, path)
    src = src_text.encode("utf-8", "replace")
    try:
        tree = parser(g).parse(src)
    except Exception:
        return None
    lines = [l + "\n" for l in src_text.split("\n")]  # rows as tree-sitter and git count them ("\n" only)
    funcs, conts = FUNC[g], CONTAINER.get(g, set())
    found = []  # (qual, container, start_row, end_row, col, kind)

    stack = [(tree.root_node, None)]  # iterative walk in source order (no recursion limit on deep trees)
    while stack:
        node, container = stack.pop()
        t = node.type
        if t in funcs:
            name = _name(node, g, src)
            if name:
                found.append((name, container, node.start_point[0], node.end_point[0], node.start_point[1],
                              "method" if container else "function"))
            continue  # nested functions belong to this unit
        if g in ("javascript", "typescript", "tsx") and t == "variable_declarator":
            v = node.child_by_field_name("value")
            n = node.child_by_field_name("name")
            if v is not None and v.type in VALUE_FUNC and n is not None and n.type == "identifier":
                decl = node.parent if node.parent is not None else node  # the full "const f = ..." statement
                found.append((_ident(n, src), container, decl.start_point[0], decl.end_point[0], decl.start_point[1],
                              "function"))
                continue
        if t in conts:
            container = _container_name(node, src) or container
        stack.extend((ch, container) for ch in reversed(node.children))
    out, seen = [], {}
    for name, cont, s, e, col, kind in found:
        qual = f"{cont}.{name}" if cont else name
        seen[qual] = seen.get(qual, 0) + 1
        uid = f"{path}:{qual}" + (f"#{seen[qual]}" if seen[qual] > 1 else "")
        code = "".join(lines[s:e + 1]).rstrip("\n")
        head = f"{path}/{cont}/{name}" if cont else f"{path}/{name}"
        text = f"{head}\n" + (f"class {cont}:" if cont else "") + code
        if len(text) > CHAR_CAP:
            text = text[:CHAR_CAP]
        first = lines[s] if s < len(lines) else ""
        indent = len(first) - len(first.lstrip(" \t"))  # indentation of the first line (patch insert rule)
        out.append((Unit(uid, kind, s + 1, e + 1, indent, None), text))
    return out
