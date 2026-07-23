# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_coverage, skip_for_quasar
from helpers.logger import logger
from helpers.perf import PerfConfig
from helpers.profiler import EntryType, Profiler, ProfilerFullMarker
from helpers.test_config import BuildMode, TestConfig
from ttexalens.tt_exalens_lib import read_words_from_device

EMITTED_TSDATA = 600
EMITTED_ZONES = 600
EMITTED_TS = 800

EXPECTED_RETAINED_TSDATA = 255
EXPECTED_RETAINED_ZONE = 255
EXPECTED_RETAINED_TS = 509


def _run(source: str) -> Profiler:
    config = PerfConfig(source)
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()
    return Profiler.get_data(
        config.test_name, config.variant_id, TestConfig.TENSIX_LOCATION
    )


@skip_for_coverage
@skip_for_quasar
def test_profiler_buffer_overflow_stress():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    runtime = _run("sources/profiler_stress_test.cpp")

    for thread_name, view in (
        ("unpack", runtime.unpack()),
        ("math", runtime.math()),
        ("pack", runtime.pack()),
    ):
        data_values = [int(v) for v in view.timestamps().marker("STRESS").raw()["data"]]
        retained = len(data_values)

        logger.info(
            "[stress tsdata] thread={} emitted={} retained={} dropped={}",
            thread_name,
            EMITTED_TSDATA,
            retained,
            EMITTED_TSDATA - retained,
        )

        assert (
            retained == EXPECTED_RETAINED_TSDATA
        ), f"[{thread_name}] expected {EXPECTED_RETAINED_TSDATA} retained, got {retained}"

        assert data_values == list(range(retained)), (
            f"[{thread_name}] retained entries are not a clean contiguous prefix; "
            f"start={data_values[:10]}, end={data_values[-5:]}"
        )


@skip_for_coverage
@skip_for_quasar
def test_profiler_buffer_overflow_zones():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    runtime = _run("sources/profiler_stress_zones_test.cpp")

    zones = runtime.unpack().zones().marker("STRESS_ZONE").frame()
    retained = len(zones)

    logger.info(
        "[stress zones] emitted={} retained={} dropped={}",
        EMITTED_ZONES,
        retained,
        EMITTED_ZONES - retained,
    )

    assert (
        retained == EXPECTED_RETAINED_ZONE
    ), f"expected {EXPECTED_RETAINED_ZONE} retained zones, got {retained}"

    assert (
        zones["duration"].notna().all()
    ), "a retained zone has no matching END (half-open zone leaked through)"
    assert (
        zones["duration"] >= 0
    ).all(), "a zone END precedes its START (broken pairing)"


@skip_for_coverage
@skip_for_quasar
def test_profiler_buffer_overflow_timestamps():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    runtime = _run("sources/profiler_stress_timestamps_test.cpp")

    ts = runtime.unpack().timestamps().marker("STRESS_TS").raw()
    timestamps = [int(v) for v in ts["timestamp"]]
    retained = len(timestamps)

    logger.info(
        "[stress ts] emitted={} retained={} dropped={}",
        EMITTED_TS,
        retained,
        EMITTED_TS - retained,
    )

    assert (
        retained == EXPECTED_RETAINED_TS
    ), f"expected {EXPECTED_RETAINED_TS} retained timestamps, got {retained}"

    assert timestamps == sorted(timestamps) and len(set(timestamps)) == len(
        timestamps
    ), "timestamps are not strictly increasing (bad decode or ordering)"


@skip_for_coverage
@skip_for_quasar
@pytest.mark.xfail(
    reason=(
        "KNOWN BUG (findings sec 6.1): is_buffer_full() reserves only 1 word per open "
        "zone, but each ZONE_END is 2 words and the destructor writes it unconditionally, "
        "so closing deeply-nested zones near a full buffer overruns write_idx into the "
        "neighbor thread's buffer. Reproduced on Wormhole 2026-07-22 "
        "(math word0 came back as ZONE_END)"
    ),
    strict=False,
)
def test_profiler_buffer_overrun_into_neighbor():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig("sources/profiler_stress_overrun_test.cpp")
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()

    # reading unpack's buffer over the NoC flushes its data cache to L1, so the spill is
    # visible when we read the math buffer next.
    read_words_from_device(
        addr=TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[0],
        word_count=0x400,
        location=TestConfig.TENSIX_LOCATION,
    )

    words = read_words_from_device(
        addr=TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[1],
        word_count=16,
        location=TestConfig.TENSIX_LOCATION,
    )

    entries = []
    i = 0
    while i < len(words):
        word = int(words[i])
        if not (word & Profiler.ENTRY_EXISTS_BIT):
            break
        kind = (word & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
        marker_id = (word & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
        entries.append((i, marker_id))
        i += 4 if kind == EntryType.TIMESTAMP_DATA.value else 2

    word0_kind = (int(words[0]) & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
    kernel_id = (int(words[0]) & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
    foreign = [(idx, mid) for idx, mid in entries if mid != kernel_id]

    assert (
        word0_kind == EntryType.ZONE_START.value and not foreign
    ), f"math buffer corrupted: word0 kind=0x{word0_kind:x}, foreign entries={foreign}"


@skip_for_coverage
@skip_for_quasar
@pytest.mark.xfail(
    reason=(
        "CONSEQUENCE of the overrun bug (findings sec 6.1): the overrun corrupts the "
        "neighbor buffer with a stray ZONE_END, so a NORMAL Profiler.get_data() parse of "
        "the (innocent) math thread raises 'Possible buffer corruption' -- the whole perf "
        "run dies. Reproduced on Wormhole. Remove this marker once the reservation is fixed."
    ),
    strict=False,
)
def test_profiler_overrun_crashes_normal_read():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig("sources/profiler_stress_overrun_test.cpp")
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()

    runtime = Profiler.get_data(
        config.test_name, config.variant_id, TestConfig.TENSIX_LOCATION
    )

    math_markers = set(str(m) for m in runtime.math().raw()["marker"])
    assert {"NEST", "FILLER"}.isdisjoint(
        math_markers
    ), f"math buffer contains unpack's markers {math_markers}"


@skip_for_coverage
@skip_for_quasar
def test_marker_hash_collision_detection_and_ceiling():
    marker = "LLK_PROFILER:foo.cpp:10:MARK"
    h = Profiler._hash_meta(marker)
    assert h == Profiler._hash_meta(marker), "hash is not deterministic"
    assert 0 <= h <= 0xFFFF, f"hash {h} is not a 16-bit value"

    m1 = ProfilerFullMarker(marker="A", file="f.cpp", line=1, id=0x1234)
    m2 = ProfilerFullMarker(marker="B", file="f.cpp", line=2, id=0x1234)
    with pytest.raises(AssertionError, match="collision"):
        Profiler._assert_no_collision({m1.id: m1}, m2)
    Profiler._assert_no_collision({m1.id: m1}, m1)

    seen: dict[int, str] = {}
    collision_at = None
    for i in range(5000):
        s = f"LLK_PROFILER:gen.cpp:{i}:MARK"
        hid = Profiler._hash_meta(s)
        if hid in seen and seen[hid] != s:
            collision_at = i
            break
        seen[hid] = s

    assert (
        collision_at is not None
    ), "expected a 16-bit hash collision within 5000 markers"
