# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
from conftest import skip_for_coverage, skip_for_quasar
from helpers.logger import logger
from helpers.perf import PerfConfig
from helpers.profiler import Profiler
from helpers.test_config import BuildMode, TestConfig

STRESS_EVENT_COUNT = 600
EXPECTED_RETAINED = 255


@skip_for_coverage
@skip_for_quasar
def test_profiler_buffer_overflow_stress():
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    configuration = PerfConfig("sources/profiler_stress_test.cpp")
    configuration.generate_variant_hash()
    configuration.build_elfs()
    configuration.run_elf_files()

    runtime = Profiler.get_data(
        configuration.test_name, configuration.variant_id, TestConfig.TENSIX_LOCATION
    )

    stress = runtime.unpack().timestamps().marker("STRESS").raw()
    data_values = [int(v) for v in stress["data"]]
    retained = len(data_values)

    logger.info(
        "[profiler stress] emitted={} retained={} dropped={}",
        STRESS_EVENT_COUNT,
        retained,
        STRESS_EVENT_COUNT - retained,
    )

    assert retained == EXPECTED_RETAINED, (
        f"Buffer capacity changed: expected {EXPECTED_RETAINED} retained out of "
        f"{STRESS_EVENT_COUNT} emitted, got {retained}. If this is intentional "
        "(buffer size / guard / entry layout changed), update EXPECTED_RETAINED."
    )

    assert data_values == list(
        range(retained)
    ), f"Retained STRESS entries are not a clean contiguous prefix; got start={data_values[:10]}, end={data_values[-5:]}"
