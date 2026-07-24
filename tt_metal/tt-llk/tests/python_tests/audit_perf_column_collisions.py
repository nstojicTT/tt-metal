# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Audit perf-CSV column-name collisions from source — proof, in one place.

Pure Python, no hardware. Reads the real param classes and every test/perf
call site and, for each column-name hazard, proves from source:

  * WHY it is a hazard    -- the classes that emit the same column name and the
                             different C++ macros they generate;
  * HOW BAD it is         -- ACTIVE (two emitters land in the SAME variant list,
                             so one CSV gets a duplicate column and a name-keyed
                             read silently drops one) vs LATENT (one emitter per
                             CSV, but the name means different things across
                             tests) vs DEAD (class never instantiated);
  * WHAT THE FIX COSTS    -- whether each emitter is passed positionally (field
                             rename is free) or by keyword (rename touches call
                             sites).

Run from tests/python_tests/:
    python audit_perf_column_collisions.py
Exit code is 0 always; this is a report, not a gate (the gate lives in
test_analyze_perf_columns.py).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import analyze_perf_columns as apc

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "helpers" / "test_variant_parameters.py"
# Files that DEFINE params or are tooling, not call sites.
SKIP = {
    "test_variant_parameters.py",
    "analyze_perf_columns.py",
    Path(__file__).name,
    "test_analyze_perf_columns.py",
}


def _call_sites(classes: set[str]):
    """Scan every .py under python_tests for calls to the given param classes.

    Returns:
      calls   : class -> list of (file, line, {keyword arg names})
      cooccur : field-agnostic list of (file, line, [classes]) for every list
                literal (runtimes=[...]/templates=[...]) holding >=2 of them.
    """
    calls: dict[str, list] = defaultdict(list)
    cooccur: list = []
    for path in sorted(HERE.rglob("*.py")):
        if path.name in SKIP:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(HERE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in classes:
                    kw = {k.arg for k in node.keywords if k.arg}
                    calls[node.func.id].append((str(rel), node.lineno, kw))
            if isinstance(node, ast.List):
                members = [
                    el.func.id
                    for el in node.elts
                    if isinstance(el, ast.Call)
                    and isinstance(el.func, ast.Name)
                    and el.func.id in classes
                ]
                if len(members) >= 2:
                    cooccur.append((str(rel), node.lineno, members))
    return calls, cooccur


def main() -> int:
    f2c, c2m, c2f = apc.parse_param_classes([PARAMS])

    # Collisions: one field name, >1 class, >1 distinct macro.
    collisions: dict[str, set[str]] = {}
    for field, classes in f2c.items():
        macros = set().union(*(c2m.get(c, set()) for c in classes))
        if len(classes) > 1 and len(macros) > 1:
            collisions[field] = classes

    # Merges: >1 field name resolving to one macro.
    field_macro: dict[str, set[str]] = defaultdict(set)
    for cls, fields in c2f.items():
        for f in fields:
            m = apc.per_field_macro(f, cls, c2m, c2f)
            if m:
                field_macro[f].add(m)
    macro_fields: dict[str, set[str]] = defaultdict(set)
    for f, ms in field_macro.items():
        for m in ms:
            macro_fields[m].add(f)
    merges = {m: fs for m, fs in macro_fields.items() if len(fs) > 1}

    # Gather every class we need call-site data for.
    watch: set[str] = set()
    for classes in collisions.values():
        watch |= classes
    for m, fs in merges.items():
        for f in fs:
            watch |= f2c.get(f, set())
    calls, cooccur = _call_sites(watch)

    def emitter(field, cls):
        macro = sorted(c2m.get(cls, {"?"}))
        sites = calls.get(cls, [])
        n = len(sites)
        kw = any(field in s[2] for s in sites)
        how = "unused (DEAD)" if n == 0 else ("keyword" if kw else "positional")
        return macro, n, how

    print("=" * 74)
    print("FACT 0  A perf-CSV column header IS the dataclass field name")
    print("=" * 74)
    print("  helpers/perf.py:582  _dataclass_name_and_values():")
    print("      return [(f.name, getattr(obj, f.name)) for f in fields(obj)]")
    print(
        "  helpers/perf.py:735-744  names.append(name) -> pd.DataFrame(columns=names)"
    )
    print("  => two params with the same field name produce two identically-named")
    print("     columns in one CSV; a name-keyed read keeps only one.\n")

    print("=" * 74)
    print(f"FACT 1  HEADER COLLISIONS  ({len(collisions)} found)")
    print("=" * 74)
    for field in sorted(collisions):
        classes = sorted(collisions[field])
        # Active if two emitters of THIS field share a list literal.
        active = [
            (f, ln, [m for m in mem if m in classes])
            for (f, ln, mem) in cooccur
            if len(set(mem) & set(classes)) >= 2
        ]
        verdict = (
            "ACTIVE (duplicate column in one CSV)"
            if active
            else "latent (cross-test ambiguity)"
        )
        dead = [c for c in classes if not calls.get(c)]
        if dead and len(classes) - len(dead) <= 1:
            verdict = f"DEAD ({', '.join(dead)} never instantiated)"
        print(f"\n  column '{field}'  ->  {verdict}")
        for c in classes:
            macro, n, how = emitter(field, c)
            print(
                f"      {c:<22} emits {'|'.join(macro):<26} {n:>2} call sites  [{how}]"
            )
        for f, ln, mem in active:
            print(f"      ^ SAME variant list: {f}:{ln}  ({' + '.join(mem)})")

    print("\n" + "=" * 74)
    print(f"FACT 2  MERGE CANDIDATES  ({len(merges)} found: same macro, two names)")
    print("=" * 74)
    for macro in sorted(merges):
        print(f"\n  macro {macro}  <-  {', '.join(sorted(merges[macro]))}")
        for field in sorted(merges[macro]):
            for c in sorted(f2c.get(field, set())):
                _, n, how = emitter(field, c)
                print(f"      {field:<20} via {c:<22} {n:>2} call sites  [{how}]")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  header collisions : {sorted(collisions)}")
    print(f"  merge candidates  : { {m: sorted(fs) for m, fs in merges.items()} }")
    print("  Only collisions with an ACTIVE line above lose data today; the rest are")
    print("  cross-test ambiguities. Positional emitters rename for free; keyword")
    print("  emitters (e.g. MATH_OP(mathop=...)) would touch call sites, so rename the")
    print(
        "  positional siblings instead and leave the keyword one as the canonical name."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
