"""Quick CPU tests (no network, no models): python -I -m pytest -q tests  (or python tests/test_units_patch.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from issueloc.patch import group_by_path, modified_units, parse_diff, related  # noqa: E402
from issueloc.units import parse_units  # noqa: E402

SRC = '''import os

class A(Base):
    """Doc."""
    x = 1

    @property
    def p(self):
        return 1

    def m(self):
        def inner():
            return 2
        return inner()

def f(a):
    return a + 1

if True:
    def g():
        pass
'''


def units():
    return parse_units(SRC, "pkg/mod.py")


def test_ids():
    ids = [u.uid for u in units()]
    assert ids == ["pkg/mod.py:A", "pkg/mod.py:A.p", "pkg/mod.py:A.m", "pkg/mod.py:f", "pkg/mod.py:g"], ids
    p = [u for u in units() if u.uid.endswith("A.p")][0]
    assert p.start == 7  # decorator line included


def _patch(body):
    return "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n" + body


def test_change_in_method():
    pt = _patch("@@ -12,3 +12,3 @@\n         def inner():\n-            return 2\n+            return 3\n         return inner()\n")
    assert modified_units(pt, group_by_path(units())) == ["pkg/mod.py:A.m"]


def test_class_level_change_and_added_method():
    pt = _patch("@@ -4,2 +4,2 @@\n     \"\"\"Doc.\"\"\"\n-    x = 1\n+    x = 2\n")
    assert modified_units(pt, group_by_path(units())) == ["pkg/mod.py:A"]
    # append a method after A.m (old line 14 is the last line of A.m)
    pt = _patch("@@ -14,2 +14,5 @@\n         return inner()\n+\n+    def n(self):\n+        return 0\n \n")
    assert modified_units(pt, group_by_path(units())) == ["pkg/mod.py:A"]


def test_insert_in_function_body_and_module_level():
    pt = _patch("@@ -17,1 +17,2 @@\n def f(a):\n+    a = int(a)\n")
    assert modified_units(pt, group_by_path(units())) == ["pkg/mod.py:f"]
    pt = _patch("@@ -1,1 +1,2 @@\n import os\n+import sys\n")
    assert modified_units(pt, group_by_path(units())) == []


def test_parse_diff_line_numbers():
    d = parse_diff(_patch("@@ -10,4 +10,4 @@\n a\n-b\n+c\n d\n"))
    assert d["pkg/mod.py"]["changed"] == {11}


def test_body_line_looking_like_header():
    pt = _patch("@@ -16,2 +16,2 @@\n def f(a):\n---    return a + 1\n+++    return a + 2\n")
    d = parse_diff(pt)
    assert list(d) == ["pkg/mod.py"] and d["pkg/mod.py"]["changed"] == {17}
    assert modified_units(pt, group_by_path(units())) == ["pkg/mod.py:f"]


def test_two_files():
    pt = (_patch("@@ -17,1 +17,1 @@\n-def f(a):\n+def f(b):\n")
          + "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
    d = parse_diff(pt)
    assert d["pkg/mod.py"]["changed"] == {17} and d["x.py"]["changed"] == {1}


def test_related():
    assert related("a.py:A", "a.py:A.m") and related("a.py:A.m", "a.py:A")
    assert not related("a.py:A", "a.py:AB.m") and not related("a.py:A", "b.py:A.m")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("ok")
