# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
from collections import defaultdict
from pathlib import Path

import analyze_perf_columns as apc
import pytest


# These are host-only unit tests -- no device, no perf run. Override conftest's
# module-scoped report fixtures so this module emits no perf CSVs into perf_data.
@pytest.fixture(autouse=True)
def perf_report():
    yield None


@pytest.fixture(autouse=True)
def counter_report():
    yield None


# A self-contained set of param classes exercising every resolution branch:
#   THROTTLE_LEVEL  field != macro but upper-cases to it   -> confident
#   TILE_DIM_A/B    two names, one CT_DIM macro            -> align/merge candidate
#   OPX/OPY         one field "mode", two macros           -> collision
#   RTP             RuntimeParameter subclass              -> confident
#   MULTI           two fields, one macro                  -> convention review
#   NotAParam       not a param base                       -> ignored
#   DUP (x2)        duplicate class name                   -> last definition wins
PARAMS_SRC = """
from dataclasses import dataclass

class TemplateParameter: pass
class RuntimeParameter: pass

@dataclass
class THROTTLE_LEVEL(TemplateParameter):
    throttle_level: int = 0
    def convert_to_cpp(self):
        return f"constexpr int THROTTLE_LEVEL = {self.throttle_level};"

@dataclass
class TILE_DIM_A(TemplateParameter):
    tile_dim: int = 1
    def convert_to_cpp(self):
        return f"constexpr int CT_DIM = {self.tile_dim};"

@dataclass
class TILE_DIM_B(TemplateParameter):
    ct_dim_alt: int = 1
    def convert_to_cpp(self):
        return f"constexpr int CT_DIM = {self.ct_dim_alt};"

@dataclass
class OPX(TemplateParameter):
    mode: str = "x"
    def convert_to_cpp(self):
        return "#define MODE_X"

@dataclass
class OPY(TemplateParameter):
    mode: str = "y"
    def convert_to_cpp(self):
        return "#define MODE_Y"

@dataclass
class RTP(RuntimeParameter):
    rt: int = 0
    def convert_to_cpp(self):
        return f"constexpr int RT = {self.rt};"

@dataclass
class MULTI(TemplateParameter):
    a: int = 0
    b: int = 0
    def convert_to_cpp(self):
        return f"constexpr int SOMEMACRO = {self.a};"

class NotAParam:
    z: int = 0
    def convert_to_cpp(self):
        return "constexpr int NOPE = 0;"

@dataclass
class DUP(TemplateParameter):
    dup: int = 0
    def convert_to_cpp(self):
        return "#define DUP_FIRST"

@dataclass
class DUP(TemplateParameter):
    dup: int = 0
    def convert_to_cpp(self):
        return "#define DUP_SECOND"
"""


@pytest.fixture
def params_file(tmp_path):
    p = tmp_path / "params.py"
    p.write_text(PARAMS_SRC)
    return p


@pytest.fixture
def parsed(params_file):
    return apc.parse_param_classes([params_file])


@pytest.fixture
def perf_data(tmp_path):
    """Three perf CSVs (one nested) plus derived .post/.counters that must be skipped."""
    d = tmp_path / "perf_data"
    (d / "nested").mkdir(parents=True)
    (d / "test_alpha.csv").write_text(
        "marker,formats.input_A,dest_acc,throttle_level,tile_dim,mode,mean(cycles),std(cycles)\n"
        "ZONE,Float16,false,0,1,x,10,1\n"
    )
    (d / "test_beta.csv").write_text(
        "marker,ct_dim_alt,mode,run_index,mean(cycles)\nZONE,2,y,0,20\n"
    )
    (d / "nested" / "test_gamma.csv").write_text(
        "marker,throttle_level,mean(cycles)\nZONE,3,30\n"
    )
    (d / "test_alpha.post.csv").write_text("skip,me\n")
    (d / "test_alpha.counters.csv").write_text("skip,me\n")
    return d


# ── Column classification: every column lands on the right side ────────


@pytest.mark.parametrize(
    "col",
    [
        "marker",
        "formats.input_A",
        "formats.input_B",
        "formats.output",
        "unpack_to_dest",
        "dest_acc",
        "run_index",
        "test_name",
    ],
)
def test_fixed_columns_are_not_parameters(col):
    assert apc.is_parameter_column(col) is False


@pytest.mark.parametrize(
    "col",
    [
        "mean(cycles)",
        "std(cycles)",
        "cycles_mean(x)",
        "cycles_std(x)",
        "foo_pct",
        "TEXT_SIZE(trisc0)",
    ],
)
def test_metric_columns_are_not_parameters(col):
    assert apc.is_parameter_column(col) is False


@pytest.mark.parametrize("col", ["throttle_level", "tile_dim", "mode", "ct_dim_alt"])
def test_swept_columns_are_parameters(col):
    assert apc.is_parameter_column(col) is True


@pytest.mark.parametrize(
    "col", ["mean_value", "std", "means", "pct_done", "TEXT_SIZED"]
)
def test_metric_lookalikes_are_still_parameters(col):
    # Names that merely contain mean/std/pct/TEXT_SIZE but lack the '(' / '_pct'
    # boundary must NOT be swallowed as metric columns, or real params vanish.
    assert apc.is_parameter_column(col) is True


# ── Param-class parsing: nothing extracted wrong, nothing dropped ──────


def test_parse_extracts_all_fields_and_macros(parsed):
    field_to_classes, class_to_macros, class_to_fields = parsed

    assert field_to_classes["throttle_level"] == {"THROTTLE_LEVEL"}
    assert field_to_classes["mode"] == {"OPX", "OPY"}
    assert class_to_macros["THROTTLE_LEVEL"] == {"THROTTLE_LEVEL"}
    assert class_to_macros["TILE_DIM_A"] == {"CT_DIM"}
    assert class_to_macros["OPX"] == {"MODE_X"}
    assert class_to_macros["RTP"] == {"RT"}
    assert class_to_fields["MULTI"] == ["a", "b"]


def test_parse_ignores_non_param_classes(parsed):
    field_to_classes, class_to_macros, _ = parsed
    assert "z" not in field_to_classes
    assert "NotAParam" not in class_to_macros


def test_parse_duplicate_class_last_definition_wins(parsed):
    _, class_to_macros, _ = parsed
    assert class_to_macros["DUP"] == {"DUP_SECOND"}


def test_parse_skips_malformed_file(tmp_path, params_file):
    bad = tmp_path / "broken.py"
    bad.write_text("class Oops(TemplateParameter:\n    this is not python\n")
    # A syntactically broken file must be skipped, not abort the whole parse.
    field_to_classes, _, _ = apc.parse_param_classes([bad, params_file])
    assert "throttle_level" in field_to_classes


# ── Macro resolution: the align/collision signal must be exact ─────────


def test_resolve_upper_case_match_is_confident(parsed):
    f2c, c2m, c2f = parsed
    macro, confident, _ = apc.resolve_macro(
        "throttle_level", f2c["throttle_level"], c2m, c2f
    )
    assert (macro, confident) == ("THROTTLE_LEVEL", True)


def test_resolve_single_field_class_is_confident(parsed):
    f2c, c2m, c2f = parsed
    # tile_dim never matches the CT_DIM name, but its class is single-field/single-macro.
    macro, confident, _ = apc.resolve_macro("tile_dim", f2c["tile_dim"], c2m, c2f)
    assert (macro, confident) == ("CT_DIM", True)


def test_resolve_ambiguous_field_is_not_confident(parsed):
    f2c, c2m, c2f = parsed
    macro, confident, all_macros = apc.resolve_macro("mode", f2c["mode"], c2m, c2f)
    assert macro is None and confident is False
    assert all_macros == {"MODE_X", "MODE_Y"}


def test_resolve_multifield_field_is_unresolved(parsed):
    f2c, c2m, c2f = parsed
    macro, confident, _ = apc.resolve_macro("a", f2c["a"], c2m, c2f)
    assert macro is None and confident is False


# ── CSV discovery: find every real CSV, skip derived, don't invent ─────


def test_find_perf_csvs_skips_derived_and_recurses(perf_data):
    csvs = apc.find_perf_csvs(perf_data)
    assert set(csvs) == {"test_alpha", "test_beta", "test_gamma"}
    assert csvs["test_alpha"].name == "test_alpha.csv"


def test_find_perf_csvs_same_base_collapses(tmp_path):
    # Two CSVs share a base name in different dirs -> the map keeps only one.
    # This is a real fidelity hazard, so pin the current (lossy) behaviour.
    d = tmp_path / "pd"
    (d / "a").mkdir(parents=True)
    (d / "b").mkdir(parents=True)
    (d / "a" / "dup.csv").write_text("col\n1\n")
    (d / "b" / "dup.csv").write_text("col\n2\n")
    csvs = apc.find_perf_csvs(d)
    assert list(csvs) == ["dup"]
    # sorted rglob + dict overwrite -> the lexicographically-last path survives.
    assert csvs["dup"] == max((d / "a" / "dup.csv"), (d / "b" / "dup.csv"))


def test_read_header_returns_first_row(perf_data):
    header = apc.read_header(perf_data / "test_beta.csv")
    assert header == ["marker", "ct_dim_alt", "mode", "run_index", "mean(cycles)"]


def test_read_header_empty_file_is_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert apc.read_header(empty) == []


def test_empty_csv_is_still_discovered(tmp_path):
    # An empty/truncated CSV becomes a test with zero columns rather than an error;
    # downstream must not treat "no columns" as "all columns present".
    d = tmp_path / "pd"
    d.mkdir()
    (d / "truncated.csv").write_text("")
    csvs = apc.find_perf_csvs(d)
    assert "truncated" in csvs
    assert apc.read_header(csvs["truncated"]) == []


# ── End-to-end: the emitted matrix loses no column and mislabels none ──


def test_matrix_output_is_complete_and_correct(
    tmp_path, params_file, perf_data, monkeypatch
):
    out = tmp_path / "matrix.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_perf_columns.py",
            "--perf-data",
            str(perf_data),
            "--params",
            str(params_file),
            "--tests-dir",
            str(tmp_path),  # no perf_*.py here -> only our params file
            "--csv-out",
            str(out),
        ],
    )
    assert apc.main() == 0

    with out.open(newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "column",
        "macro",
        "n_tests",
        "test_alpha",
        "test_beta",
        "test_gamma",
    ]

    body = {r[0]: r[1:] for r in rows[1:]}
    # Every swept parameter column is present -- none silently dropped.
    assert set(body) == {"ct_dim_alt", "mode", "throttle_level", "tile_dim"}
    # macro tag, count, and per-test presence marks are all correct.
    assert body["ct_dim_alt"] == ["CT_DIM", "1", "", "x", ""]
    assert body["mode"] == ["MODE_X|MODE_Y", "2", "x", "x", ""]
    assert body["throttle_level"] == ["THROTTLE_LEVEL", "2", "x", "", "x"]
    assert body["tile_dim"] == ["CT_DIM", "1", "x", "", ""]


def test_main_reports_align_and_collision(
    tmp_path, params_file, perf_data, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_perf_columns.py",
            "--perf-data",
            str(perf_data),
            "--params",
            str(params_file),
            "--tests-dir",
            str(tmp_path),
        ],
    )
    assert apc.main() == 0
    out = capsys.readouterr().out

    assert "ALIGN CANDIDATES" in out
    assert "macro CT_DIM:" in out and "ct_dim_alt" in out and "tile_dim" in out
    assert "COLLISIONS" in out
    assert "mode: classes ['OPX', 'OPY']" in out


def test_main_errors_when_no_csvs(tmp_path, params_file, monkeypatch, capsys):
    empty_pd = tmp_path / "empty_pd"
    empty_pd.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_perf_columns.py",
            "--perf-data",
            str(empty_pd),
            "--params",
            str(params_file),
            "--tests-dir",
            str(tmp_path),
        ],
    )
    assert apc.main() == 1
    assert "No perf CSVs found" in capsys.readouterr().out


def test_static_enumeration_covers_full_vocabulary(
    tmp_path, params_file, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_perf_columns.py",
            "--params",
            str(params_file),
            "--tests-dir",
            str(tmp_path),
            "--static",
        ],
    )
    assert apc.main() == 0
    out = capsys.readouterr().out

    # Every field in the vocabulary, and each report section, is present.
    for field in ["throttle_level", "tile_dim", "ct_dim_alt", "mode", "rt", "a", "b"]:
        assert field in out
    assert "MERGE CANDIDATES" in out and "macro CT_DIM:" in out
    assert "COLLISIONS" in out and "mode:" in out
    assert "CONVENTION REVIEW" in out and "MULTI:" in out


# ── Guardrail against the REAL param classes ───────────────────────────
#
# Snapshot of the perf-CSV column hazards that exist in the actual param
# classes today. These run against helpers/test_variant_parameters.py + the
# real perf_*.py, so they FAIL the moment the vocabulary drifts.
#
# HEADER COLLISION = one column name emitted by classes with different C++
# macros -> same header, different meaning -> unsafe to compare across tests.
# MERGE CANDIDATE  = two column names mapping to one macro -> the same quantity
# under two names -> should be collapsed.
KNOWN_HEADER_COLLISIONS = {"mathop", "op", "tile_cnt", "value_bits"}
KNOWN_MERGE_CANDIDATES = {"POOL_TYPE": frozenset({"pool_type", "reduce_pool_type"})}


def _real_param_maps():
    here = Path(apc.__file__).resolve().parent
    files = [here / "helpers" / "test_variant_parameters.py"] + sorted(
        here.glob("perf_*.py")
    )
    return apc.parse_param_classes(files)


def test_no_new_header_collisions():
    field_to_classes, class_to_macros, _ = _real_param_maps()
    collisions = set()
    for field, classes in field_to_classes.items():
        macros = set()
        for cls in classes:
            macros |= class_to_macros.get(cls, set())
        if len(classes) > 1 and len(macros) > 1:
            collisions.add(field)

    assert collisions == KNOWN_HEADER_COLLISIONS, (
        "perf CSV header collisions drifted from the pinned snapshot.\n"
        f"  NEW (split or rename the column): {sorted(collisions - KNOWN_HEADER_COLLISIONS)}\n"
        f"  FIXED (remove from allowlist):    {sorted(KNOWN_HEADER_COLLISIONS - collisions)}"
    )


def test_no_new_merge_candidates():
    _, class_to_macros, class_to_fields = _real_param_maps()
    field_macros: dict[str, set[str]] = defaultdict(set)
    for cls, fields in class_to_fields.items():
        for field in fields:
            macro = apc.per_field_macro(field, cls, class_to_macros, class_to_fields)
            if macro:
                field_macros[field].add(macro)

    macro_to_fields: dict[str, set[str]] = defaultdict(set)
    for field, macros in field_macros.items():
        for macro in macros:
            macro_to_fields[macro].add(field)
    merges = {m: frozenset(fs) for m, fs in macro_to_fields.items() if len(fs) > 1}

    assert merges == KNOWN_MERGE_CANDIDATES, (
        "perf CSV merge candidates drifted from the pinned snapshot.\n"
        f"  actual: { {m: sorted(fs) for m, fs in merges.items()} }"
    )
