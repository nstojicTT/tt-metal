# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Snapshot perf-CSV column hazards (collisions + merge candidates) into a bundle
you can keep after the run dir is wiped.

For every flagged column it writes small example CSVs -- the column plus 'marker'
and its distinct values -- one per emitting test, so tomorrow you can see e.g.
'mathop' meaning different things across tests, or pool_type/reduce_pool_type
being the same thing under two names. It also copies the raw CSVs and the full
analyze report, then tars the whole thing.

Usage (from tests/python_tests/, after a perf run):
    python save_column_evidence.py --out ~/perf_column_evidence
Then copy it off the machine:
    scp ~/perf_column_evidence.tar.gz you@laptop:~/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import analyze_perf_columns as apc

MAX_ROWS = 12  # sample rows per (column, test)


def _read(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def _extract(csv_path: Path, column: str, out_path: Path):
    """Write [marker, <column>] sample rows (distinct on the column) for one CSV."""
    header, data = _read(csv_path)
    if column not in header:
        return 0
    ci = header.index(column)
    mi = header.index("marker") if "marker" in header else None
    seen, out_rows = set(), []
    for r in data:
        if ci >= len(r):
            continue
        val = r[ci]
        if val in seen:
            continue
        seen.add(val)
        out_rows.append([r[mi] if mi is not None and mi < len(r) else "", val])
        if len(out_rows) >= MAX_ROWS:
            break
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker", column])
        w.writerows(out_rows)
    return len(out_rows)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perf-data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path.home() / "perf_column_evidence")
    ap.add_argument(
        "--params", type=Path, default=here / "helpers" / "test_variant_parameters.py"
    )
    ap.add_argument("--tests-dir", type=Path, default=here)
    args = ap.parse_args()

    perf_data = args.perf_data
    if perf_data is None:
        for cand in (here.parent.parent / "perf_data", here / "perf_data"):
            if cand.is_dir():
                perf_data = cand
                break
    if perf_data is None or not perf_data.is_dir():
        print("ERROR: no perf_data dir. Pass --perf-data.")
        return 1

    csvs = apc.find_perf_csvs(perf_data)
    if not csvs:
        print(f"No perf CSVs under {perf_data}.")
        return 1

    # Which tests emit each parameter column (present in real CSVs).
    col_to_tests: dict[str, list[str]] = defaultdict(list)
    for base, path in csvs.items():
        for c in apc.read_header(path):
            if apc.is_parameter_column(c):
                col_to_tests[c].append(base)

    # Resolve columns to macros using the param definitions.
    param_files = [args.params] + sorted(args.tests_dir.glob("perf_*.py"))
    f2c, c2m, c2f = apc.parse_param_classes(param_files)

    col_macro = {}
    for c in col_to_tests:
        col_macro[c] = apc.resolve_macro(c, f2c.get(c, set()), c2m, c2f)

    # Collisions: one column name, its classes span >1 macro AND >1 emitting class
    # actually appears in the CSVs (so we don't flag consistently-emitted names).
    collisions = {}
    for c in col_to_tests:
        classes = f2c.get(c, set())
        macros = (
            set().union(*(c2m.get(k, set()) for k in classes)) if classes else set()
        )
        if len(classes) > 1 and len(macros) > 1:
            collisions[c] = sorted(macros)

    # Align candidates: >1 confident column name resolves to the same macro.
    macro_to_cols: dict[str, set] = defaultdict(set)
    for c, (macro, confident, _) in col_macro.items():
        if macro and confident:
            macro_to_cols[macro].add(c)
    aligns = {m: sorted(cs) for m, cs in macro_to_cols.items() if len(cs) > 1}

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "collisions").mkdir(parents=True)
    (out / "align").mkdir(parents=True)
    (out / "raw_csvs").mkdir(parents=True)

    # 1) Copy every raw CSV (complete data, survives the wipe).
    for base, path in csvs.items():
        shutil.copy(path, out / "raw_csvs" / f"{base}.csv")

    # 2) Full analyze report as text.
    argv = sys.argv
    sys.argv = [
        "analyze_perf_columns.py",
        "--perf-data",
        str(perf_data),
        "--params",
        str(args.params),
        "--tests-dir",
        str(args.tests_dir),
    ]
    try:
        with (out / "analyze_report.txt").open("w") as f, contextlib.redirect_stdout(f):
            apc.main()
    finally:
        sys.argv = argv

    # 3) Per-hazard example extracts.
    summary = []
    summary.append(f"Perf column evidence  ({len(csvs)} CSVs from {perf_data})\n")
    summary.append(f"COLLISIONS (same header, different meaning): {sorted(collisions)}")
    for c in sorted(collisions):
        d = out / "collisions" / c
        d.mkdir(exist_ok=True)
        summary.append(f"\n[{c}]  macros seen in definitions: {collisions[c]}")
        for base in sorted(set(col_to_tests[c])):
            n = _extract(csvs[base], c, d / f"{base}.csv")
            summary.append(
                f"    {base:<32} {n} distinct value(s) -> collisions/{c}/{base}.csv"
            )

    summary.append(
        f"\n\nALIGN CANDIDATES (same macro, different names -> duplicates): "
        f"{ {m: cs for m, cs in aligns.items()} }"
    )
    for macro, cols in sorted(aligns.items()):
        d = out / "align" / macro
        d.mkdir(exist_ok=True)
        summary.append(f"\n[{macro}]  columns: {cols}")
        for c in cols:
            for base in sorted(set(col_to_tests[c])):
                n = _extract(csvs[base], c, d / f"{c}__{base}.csv")
                summary.append(
                    f"    {c:<20} in {base:<32} {n} value(s) -> align/{macro}/{c}__{base}.csv"
                )

    # Unresolved (name != macro) columns, for reference.
    unresolved = sorted(c for c, (m, conf, allm) in col_macro.items() if not m)
    summary.append(
        f"\n\nUNRESOLVED name!=macro columns (naming hygiene, not duplicates): {unresolved}"
    )

    (out / "SUMMARY.txt").write_text("\n".join(summary) + "\n")

    # 4) Tar it up next to the folder.
    tar_path = out.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out, arcname=out.name)

    print("\n".join(summary))
    print(f"\nBundle:  {out}")
    print(f"Tarball: {tar_path}")
    print(f"\nCopy it off the machine, e.g.:\n    scp {tar_path} <you>@<laptop>:~/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
