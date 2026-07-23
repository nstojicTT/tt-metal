# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Consolidate and align perf-test CSV headers across ALL LLK perf tests.

Run this AFTER a full perf run has generated the combined CSVs
(``<tt-llk-root>/perf_data/<base>/<base>.csv``). Analysis needs no hardware.

What it does
------------
1. Reads the header row of every generated perf CSV.
2. Classifies each header as a swept PARAMETER column (the ones that drift)
   vs. a systematic column (marker / mean(...) / std(...) / TEXT_SIZE(...) /
   the fixed formats.*/dest_acc/unpack_to_dest columns / counter metrics).
3. Resolves each parameter column back to the param class(es) that emit it and,
   where it can be done unambiguously, to the C++ macro that class generates
   (parsed from ``convert_to_cpp`` in test_variant_parameters.py + any
   file-local param classes in perf_*.py).
4. Prints:
   - a COLUMN x TEST presence matrix (every parameter column, which tests emit it);
   - ALIGN CANDIDATES: different header names that resolve to the SAME C++ macro
     (provably the same thing -> can collapse to one column);
   - COLLISIONS: one header name emitted by classes with DIFFERENT macros
     (same name, different meaning -> must be split);
   - a per-test column listing.
   Optionally writes the matrix to --csv-out for a spreadsheet.

Usage
-----
    python analyze_perf_columns.py \
        --perf-data /path/to/tt-llk/perf_data \
        --params    helpers/test_variant_parameters.py \
        --tests-dir . \
        --csv-out   perf_column_matrix.csv

All arguments have sensible defaults when run from tests/python_tests/.
Macro resolution for multi-field classes (e.g. CRK_TILE_DIMM: c_dimm->CT_DIM)
is best-effort and flagged as low-confidence; the presence matrix is exact.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

# ── Column classification ──────────────────────────────────────────────

# Systematic (non-parameter) columns: present by construction, already aligned.
_FIXED_COLUMNS = {
    "marker",
    "formats.input_A",
    "formats.input_B",
    "formats.output",
    "unpack_to_dest",
    "dest_acc",
    "run_index",
    "test_name",  # perf_fused.py's manual sweep column
}
# Metric / stat / code-size columns (systematic, generated from run types).
_METRIC_RE = re.compile(r"(^|_)(mean|std)\(|_pct\b|^TEXT_SIZE\(")


def is_parameter_column(col: str) -> bool:
    """True if a header is a swept parameter column (the drift-prone kind)."""
    if col in _FIXED_COLUMNS:
        return False
    if _METRIC_RE.search(col):
        return False
    return True


# ── Parse param classes: field names + emitted C++ macro(s) ────────────

_DEFINE_RE = re.compile(r"#define\s+([A-Za-z_]\w*)")
# LHS identifier of a `constexpr <type...> NAME =` declaration.
_CONSTEXPR_RE = re.compile(r"constexpr\s+[^;=]*?([A-Za-z_]\w*)\s*=")


def _static_text(node: ast.AST) -> str:
    """Reconstruct the static (non-interpolated) text of a string/f-string node."""
    parts: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.append(sub.value)
    return "\n".join(parts)


def _macros_from_convert(func: ast.FunctionDef) -> set[str]:
    macros: set[str] = set()
    text = _static_text(func)
    macros.update(_DEFINE_RE.findall(text))
    macros.update(_CONSTEXPR_RE.findall(text))
    return macros


def _field_names(cls: ast.ClassDef) -> list[str]:
    """Top-level annotated dataclass fields (public, non-callable)."""
    names = []
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            if not name.startswith("_"):
                names.append(name)
    return names


def _is_param_class(cls: ast.ClassDef) -> bool:
    base_names = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    return bool(base_names & {"TemplateParameter", "RuntimeParameter"})


def parse_param_classes(paths: list[Path]):
    """Return (field_to_classes, class_to_macros, class_to_fields)."""
    field_to_classes: dict[str, set[str]] = defaultdict(set)
    class_to_macros: dict[str, set[str]] = {}
    class_to_fields: dict[str, list[str]] = {}

    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if not _is_param_class(cls):
                continue
            fields = _field_names(cls)
            macros: set[str] = set()
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == "convert_to_cpp":
                    macros = _macros_from_convert(item)
            # Last definition wins on duplicate class names (matches Python).
            class_to_fields[cls.name] = fields
            class_to_macros[cls.name] = macros
            for f in fields:
                field_to_classes[f].add(cls.name)

    return field_to_classes, class_to_macros, class_to_fields


def resolve_macro(field: str, classes: set[str], class_to_macros, class_to_fields):
    """Best-effort field -> macro. Returns (macro_or_None, confident, all_macros)."""
    all_macros: set[str] = set()
    for c in classes:
        all_macros |= class_to_macros.get(c, set())

    # Exact match: a class emits a macro named exactly (or upper-cased) like the field.
    if field in all_macros:
        return field, True, all_macros
    up = field.upper()
    if up in all_macros:
        return up, True, all_macros

    # Single-field class with exactly one macro -> unambiguous.
    single = [
        c
        for c in classes
        if len(class_to_fields.get(c, [])) == 1
        and len(class_to_macros.get(c, set())) == 1
    ]
    if len(classes) == 1 and single:
        return next(iter(class_to_macros[single[0]])), True, all_macros

    return None, False, all_macros


def per_field_macro(field: str, cls: str, class_to_macros, class_to_fields):
    """Resolve ONE field within ONE class to its macro, or None if ambiguous."""
    macros = class_to_macros.get(cls, set())
    if field in macros:
        return field
    if field.upper() in macros:
        return field.upper()
    if len(class_to_fields.get(cls, [])) == 1 and len(macros) == 1:
        return next(iter(macros))
    return None


# ── CSV discovery + header extraction ──────────────────────────────────


def find_perf_csvs(perf_data_dir: Path) -> dict[str, Path]:
    """Map base test name -> primary combined CSV (skip .post/.counters)."""
    out: dict[str, Path] = {}
    for p in sorted(perf_data_dir.rglob("*.csv")):
        name = p.name
        if name.endswith(".post.csv") or name.endswith(".counters.csv"):
            continue
        base = name[:-4]
        out[base] = p
    return out


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        row = next(csv.reader(f), [])
    return row


# ── Static enumeration (no CSVs / no hardware) ─────────────────────────


def run_static(field_to_classes, class_to_macros, class_to_fields) -> int:
    """Enumerate EVERY possible parameter column and every misalignment from the
    param definitions alone. Answers 'what is the full column vocabulary and
    where can it drift?' without needing a perf run."""
    # Per (class, field) macro resolution.
    field_macros: dict[str, set[str]] = defaultdict(set)  # field -> {macros}
    unresolved: list[tuple[str, str, set[str]]] = []  # (class, field, class macros)
    for cls, fields in sorted(class_to_fields.items()):
        for f in fields:
            m = per_field_macro(f, cls, class_to_macros, class_to_fields)
            if m:
                field_macros[f].add(m)
            else:
                unresolved.append((cls, f, class_to_macros.get(cls, set())))

    all_fields = sorted(field_to_classes)
    print(
        f"\nParameter classes: {len(class_to_fields)}   "
        f"distinct field/column names: {len(all_fields)}\n"
    )

    print("=" * 78)
    print("FULL PARAMETER-COLUMN VOCABULARY  (field -> class(es) -> macro)")
    print("=" * 78)
    for f in all_fields:
        classes = sorted(field_to_classes[f])
        macros = sorted(field_macros.get(f, set())) or ["?"]
        print(f"  {f:<30} {'|'.join(macros):<28} {classes}")

    # MERGE candidates: one macro, multiple field names.
    macro_to_fields: dict[str, set[str]] = defaultdict(set)
    for f, macros in field_macros.items():
        for m in macros:
            macro_to_fields[m].add(f)
    print("\n" + "=" * 78)
    print("MERGE CANDIDATES  (same C++ macro, DIFFERENT field/column names)")
    print("=" * 78)
    any_merge = False
    for m, fs in sorted(macro_to_fields.items()):
        if len(fs) > 1:
            any_merge = True
            print(f"  macro {m}:  " + "  vs  ".join(sorted(fs)))
    if not any_merge:
        print("  (none)")

    # COLLISIONS: one field name, multiple classes emitting different macros.
    print("\n" + "=" * 78)
    print("COLLISIONS  (same field/column name, DIFFERENT classes/macros)")
    print("=" * 78)
    any_coll = False
    for f in all_fields:
        classes = field_to_classes[f]
        macros = set()
        for c in classes:
            macros |= class_to_macros.get(c, set())
        if len(classes) > 1 and len(macros) > 1:
            any_coll = True
            print(f"  {f}: classes {sorted(classes)} -> macros {sorted(macros)}")
    if not any_coll:
        print("  (none)")

    # CONVENTION REVIEW: multi-field classes whose fields don't map 1:1 to a macro.
    print("\n" + "=" * 78)
    print(
        "CONVENTION REVIEW  (multi-field classes; field<->macro not 1:1, judge by hand)"
    )
    print("=" * 78)
    seen = set()
    for cls, field, macros in unresolved:
        if cls in seen:
            continue
        seen.add(cls)
        print(
            f"  {cls}: fields {class_to_fields.get(cls, [])} -> macros {sorted(macros)}"
        )
    if not seen:
        print("  (none)")

    return 0


# ── Report ─────────────────────────────────────────────────────────────


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--perf-data",
        type=Path,
        default=None,
        help="Directory containing generated perf CSVs (searched recursively). "
        "Default: auto-detect ../../perf_data then ./perf_data.",
    )
    ap.add_argument(
        "--params", type=Path, default=here / "helpers" / "test_variant_parameters.py"
    )
    ap.add_argument(
        "--tests-dir",
        type=Path,
        default=here,
        help="Dir with perf_*.py (for file-local param classes)",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional: write the column x test matrix here",
    )
    ap.add_argument(
        "--static",
        action="store_true",
        help="Enumerate the full column vocabulary + all misalignments from the "
        "param definitions ALONE (no CSVs, no hardware).",
    )
    args = ap.parse_args()

    # Build the field/macro maps from the param definitions + local classes.
    param_files = [args.params] + sorted(args.tests_dir.glob("perf_*.py"))
    field_to_classes, class_to_macros, class_to_fields = parse_param_classes(
        param_files
    )

    if args.static:
        return run_static(field_to_classes, class_to_macros, class_to_fields)

    # Locate perf_data
    perf_data = args.perf_data
    if perf_data is None:
        for cand in (here.parent.parent / "perf_data", here / "perf_data"):
            if cand.is_dir():
                perf_data = cand
                break
    if perf_data is None or not perf_data.is_dir():
        print("ERROR: could not find perf_data dir. Pass --perf-data explicitly.")
        return 1

    csvs = find_perf_csvs(perf_data)
    if not csvs:
        print(f"No perf CSVs found under {perf_data}. Run the perf suite first.")
        return 1

    # column -> set(tests), plus per-test param columns
    col_to_tests: dict[str, set[str]] = defaultdict(set)
    test_to_cols: dict[str, list[str]] = {}
    for base, path in csvs.items():
        header = read_header(path)
        param_cols = [c for c in header if is_parameter_column(c)]
        test_to_cols[base] = param_cols
        for c in param_cols:
            col_to_tests[c].add(base)

    all_cols = sorted(col_to_tests)

    # Resolve each column to a macro
    col_macro: dict[str, tuple] = {}
    for c in all_cols:
        classes = field_to_classes.get(c, set())
        col_macro[c] = resolve_macro(c, classes, class_to_macros, class_to_fields)

    print(
        f"\nPerf CSVs analysed: {len(csvs)} tests, {len(all_cols)} distinct parameter columns"
    )
    print(f"(source: {perf_data})\n")

    # 1) COLUMN x TEST matrix
    print("=" * 78)
    print("PARAMETER COLUMN  ->  emitting tests   [C++ macro]")
    print("=" * 78)
    for c in all_cols:
        macro, confident, all_macros = col_macro[c]
        tag = ""
        if macro:
            tag = f"[{macro}]" if confident else f"[?{macro}]"
        elif all_macros:
            tag = f"[?{'|'.join(sorted(all_macros))}]"
        tests = sorted(col_to_tests[c])
        print(f"  {c:<32} {len(tests):>2}x  {tag}")
        print(f"       {', '.join(tests)}")

    # 2) ALIGN CANDIDATES: same macro, different header names
    macro_to_cols: dict[str, set[str]] = defaultdict(set)
    for c in all_cols:
        macro, confident, _ = col_macro[c]
        if macro and confident:
            macro_to_cols[macro].add(c)
    print("\n" + "=" * 78)
    print("ALIGN CANDIDATES  (same C++ macro, different header names -> collapse)")
    print("=" * 78)
    found = False
    for macro, cols in sorted(macro_to_cols.items()):
        if len(cols) > 1:
            found = True
            print(f"  macro {macro}:  " + "  vs  ".join(sorted(cols)))
    if not found:
        print("  (none with high confidence)")

    # 3) COLLISIONS: same header name, classes with different macros
    print("\n" + "=" * 78)
    print(
        "COLLISIONS  (one header name, multiple classes/macros -> same name, diff meaning)"
    )
    print("=" * 78)
    found = False
    for c in all_cols:
        classes = field_to_classes.get(c, set())
        macros = set()
        for cl in classes:
            macros |= class_to_macros.get(cl, set())
        if len(classes) > 1 and len(macros) > 1:
            found = True
            print(f"  {c}: classes {sorted(classes)} -> macros {sorted(macros)}")
    if not found:
        print("  (none)")

    # Optional CSV matrix
    if args.csv_out:
        tests_sorted = sorted(csvs)
        with args.csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["column", "macro", "n_tests"] + tests_sorted)
            for c in all_cols:
                macro, confident, all_macros = col_macro[c]
                mtag = macro or ("|".join(sorted(all_macros)) if all_macros else "")
                row = [c, mtag, len(col_to_tests[c])]
                row += ["x" if t in col_to_tests[c] else "" for t in tests_sorted]
                w.writerow(row)
        print(f"\nMatrix written to {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
