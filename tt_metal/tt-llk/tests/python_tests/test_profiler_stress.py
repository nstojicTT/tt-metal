# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_coverage, skip_for_quasar
from helpers.logger import logger
from helpers.perf import PerfConfig
from helpers.profiler import EntryType, Profiler
from helpers.test_config import BuildMode, TestConfig
from ttexalens.tt_exalens_lib import read_words_from_device

EMITTED_TSDATA = 600
EMITTED_ZONES = 600
EMITTED_TS = 800

EXPECTED_RETAINED_TSDATA = 255
EXPECTED_RETAINED_ZONE = 255
EXPECTED_RETAINED_TS = 509


def _run(source: str) -> Profiler:
    """Compile + run a stress kernel on device and return its parsed profiler data."""
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

        # Clean contiguous prefix 0..retained-1 == write/read fidelity + tail-drop.
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

    # Strictly increasing == correct order + sane 44-bit timestamp decode.
    assert timestamps == sorted(timestamps) and len(set(timestamps)) == len(
        timestamps
    ), "timestamps are not strictly increasing (bad decode or ordering)"


@skip_for_coverage
@skip_for_quasar
def test_profiler_buffer_overrun_into_neighbor():
    """Phase 2 — provoke the write-side reservation overrun (findings sec 6.1).

    The unpack kernel fills its buffer near-full then opens a deep nest of zones.
    is_buffer_full() reserves only 1 word per open zone, but each ZONE_END is 2
    words and the destructor writes it unconditionally, so closing the nest should
    push write_idx past the 1024-word buffer into the adjacent MATH buffer.

    Detection: read the math buffer's first word directly (bypassing the parser,
    which would raise on the corruption). A healthy math thread's first entry is its
    own KERNEL ZONE_START; if unpack overran, that word is a stray ZONE_END.

    A FAILURE here means we reproduced the overrun bug -- that is the goal of the
    hunt. Once confirmed we decide whether to fix the reservation or mark this xfail
    to document the known bug.
    """
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig("sources/profiler_stress_overrun_test.cpp")
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()

    # Read the math buffer raw. Math emitted nothing, so its first word must be its
    # own KERNEL ZONE_START unless a neighbor overran into it.
    math_addr = TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[1]
    words = read_words_from_device(
        addr=math_addr,
        word_count=8,
        location=TestConfig.TENSIX_LOCATION,
    )
    kind = (words[0] & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
    kind_name = {
        EntryType.TIMESTAMP.value: "TIMESTAMP",
        EntryType.TIMESTAMP_DATA.value: "TIMESTAMP_DATA",
        EntryType.ZONE_START.value: "ZONE_START",
        EntryType.ZONE_END.value: "ZONE_END",
    }.get(kind, f"0b{kind:04b}")

    logger.info(
        "[overrun probe] math word0=0x{:08x} kind={} ({})",
        int(words[0]),
        kind_name,
        "healthy" if kind == EntryType.ZONE_START.value else "OVERRUN from unpack!",
    )

    assert kind == EntryType.ZONE_START.value, (
        f"Neighbor (math) buffer corrupted: first word is {kind_name}, expected ZONE_START. "
        "Unpack's deeply-nested zone closes overran write_idx into the math buffer "
        "(write-side reservation shortfall, findings sec 6.1). Bug reproduced."
    )
