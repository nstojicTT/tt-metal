# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Scan GENERATED perf CSVs for duplicate column headers.

Why this exists: analyze_perf_columns.py reads param *definitions* and dedupes
headers into a set, so it cannot see a column emitted twice into ONE CSV. This
reads the raw header row of every generated CSV and reports the real thing:

  * WITHIN-FILE DUPLICATES -- the same header name appearing >1 time in one CSV.
    This is a data-loss bug: pandas/csv keyed by name keeps only one, so one
    column's values are silently unreadable (e.g. INPUT_TILE_CNT + OUTPUT_TILE_CNT
    both emit 'tile_cnt').
  * a global inventory -- every distinct column and how many CSVs carry it, so
    you can eyeball anything unexpected the static tool never sees (runtime /
    systematic / counter / cross-join columns).

Run AFTER a perf run, from tests/python_tests/:
    python find_duplicate_perf_columns.py
    python find_duplicate_perf_columns.py --perf-data /path/to/perf_data --csv-out dups.csv

Exit code: 2 if any within-file duplicate is found (so it can double as a check),
0 otherwise; 1 on setup error (no CSVs).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def find_csvs(perf_data: Path) -> list[Path]:
    """Every CSV under perf_data, tagged nowhere-special: main, .post and
    .counters are all scanned (a duplicate in any of them is a bug)."""
    return sorted(perf_data.rglob("*.csv"))


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return next(csv.reader(f), [])


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--perf-data",
        type=Path,
        default=None,
        help="Dir of generated perf CSVs (recursive). Default: auto-detect "
        "../../perf_data then ./perf_data.",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional: write the duplicate report here.",
    )
    args = ap.parse_args()

    perf_data = args.perf_data
    if perf_data is None:
        for cand in (here.parent.parent / "perf_data", here / "perf_data"):
            if cand.is_dir():
                perf_data = cand
                break
    if perf_data is None or not perf_data.is_dir():
        print(
            "ERROR: no perf_data dir. Run the perf suite first "
            "(run_llk_perf_wormhole.sh 1 1) or pass --perf-data."
        )
        return 1

    csvs = find_csvs(perf_data)
    if not csvs:
        print(f"No CSVs under {perf_data}. Run the perf suite first.")
        return 1

    # file -> {dup_name: count};  and global name -> #files it appears in
    dup_by_file: dict[str, dict[str, int]] = {}
    col_file_count: Counter = Counter()
    dup_name_files: dict[str, list[str]] = defaultdict(list)

    for path in csvs:
        header = read_header(path)
        counts = Counter(header)
        for name in set(header):
            col_file_count[name] += 1
        dups = {name: n for name, n in counts.items() if n > 1}
        if dups:
            rel = str(path.relative_to(perf_data))
            dup_by_file[rel] = dups
            for name in dups:
                dup_name_files[name].append(rel)

    print(f"Scanned {len(csvs)} CSVs under {perf_data}")
    print(f"Distinct column names across all CSVs: {len(col_file_count)}\n")

    print("=" * 76)
    print("WITHIN-FILE DUPLICATE COLUMNS  (same header >1x in one CSV = data loss)")
    print("=" * 76)
    if not dup_by_file:
        print("  none")
    else:
        for rel in sorted(dup_by_file):
            parts = ", ".join(
                f"{name} x{n}" for name, n in sorted(dup_by_file[rel].items())
            )
            print(f"  {rel}\n      {parts}")

    print("\n" + "=" * 76)
    print("DUPLICATED NAMES ROLLUP  (name -> # CSVs where it is duplicated)")
    print("=" * 76)
    if not dup_name_files:
        print("  none")
    else:
        for name in sorted(dup_name_files, key=lambda n: (-len(dup_name_files[n]), n)):
            files = dup_name_files[name]
            print(f"  {name:<28} {len(files)} file(s): {', '.join(sorted(files))}")

    if args.csv_out:
        with args.csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["csv_file", "duplicated_column", "count"])
            for rel in sorted(dup_by_file):
                for name, n in sorted(dup_by_file[rel].items()):
                    w.writerow([rel, name, n])
        print(f"\nReport written to {args.csv_out}")

    total = sum(len(d) for d in dup_by_file.values())
    print(
        f"\n{'FAIL' if total else 'OK'}: "
        f"{total} duplicate-column occurrence(s) across {len(dup_by_file)} CSV(s)."
    )
    return 2 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
