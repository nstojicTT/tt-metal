# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_coverage, skip_for_quasar
from helpers.logger import logger
from helpers.perf import PerfConfig
from helpers.profiler import Profiler
from helpers.test_config import BuildMode, TestConfig

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
