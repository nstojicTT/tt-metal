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
@pytest.mark.xfail(
    reason=(
        "KNOWN BUG (findings sec 6.1): is_buffer_full() reserves only 1 word per open "
        "zone, but each ZONE_END is 2 words and the destructor writes it unconditionally, "
        "so closing deeply-nested zones near a full buffer overruns write_idx into the "
        "neighbor thread's buffer. Reproduced on Wormhole 2026-07-22 "
        "(math word0 came back as ZONE_END). Remove this marker once the reservation is fixed."
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

    # Diagnostic: how full did UNPACK actually get, and did its zone closes cross the
    # 1024-word boundary into the math buffer? Count non-empty words in the unpack buffer
    # (final write_idx footprint) and dump the tail around the boundary. If write_idx
    # stayed <= 1024, no overrun happened and the recipe no longer stresses the guard.
    BUFFER_LENGTH = 0x400  # 1024 words/thread; mirrors profiler.h BUFFER_LENGTH
    unpack_addr = TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[0]
    unpack_words = read_words_from_device(
        addr=unpack_addr,
        word_count=BUFFER_LENGTH,
        location=TestConfig.TENSIX_LOCATION,
    )
    # Footprint = last non-zero word + 1. Reliable because init memsets the buffer to 0
    # and entries are written contiguously; a second-word timestamp can have any bits, so
    # counting the exists-bit would over-count.
    nonzero = [j for j, w in enumerate(unpack_words) if int(w) != 0]
    unpack_fill = (nonzero[-1] + 1) if nonzero else 0
    logger.info(
        "[overrun probe] unpack write_idx footprint: {} / {} words ({} free); "
        "{} boundary; tail w1012..w1023 = {}",
        unpack_fill,
        BUFFER_LENGTH,
        BUFFER_LENGTH - unpack_fill,
        "REACHED" if unpack_fill >= BUFFER_LENGTH else "did NOT reach",
        " ".join(
            f"w{1012 + j}=0x{int(w):08x}" for j, w in enumerate(unpack_words[1012:])
        ),
    )

    # Read a window of the math buffer raw. Math's kernel is empty, so a healthy math
    # buffer holds ONLY its own KERNEL zone: a ZONE_START at word 0 plus its ZONE_END,
    # every entry carrying the same KERNEL marker id. We scan the whole window (not just
    # word 0) because, depending on the exact fill level, unpack's spill can land past
    # word 0 -- word 0 may still read as a valid ZONE_START while a stray entry sits
    # deeper in. Any entry whose marker id differs from word 0's is foreign data (unpack's
    # NEST/FILLER markers) that overran write_idx into this buffer.
    WINDOW_WORDS = 16
    math_addr = TestConfig.THREAD_PERFORMANCE_DATA_BUFFER[1]
    words = read_words_from_device(
        addr=math_addr,
        word_count=WINDOW_WORDS,
        location=TestConfig.TENSIX_LOCATION,
    )

    def _kind_name(word):
        kind = (word & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
        return {
            EntryType.TIMESTAMP.value: "TIMESTAMP",
            EntryType.TIMESTAMP_DATA.value: "TIMESTAMP_DATA",
            EntryType.ZONE_START.value: "ZONE_START",
            EntryType.ZONE_END.value: "ZONE_END",
        }.get(kind, f"0b{kind:04b}")

    # Best-effort decode of the entry stream (mirrors Profiler._parse_thread strides:
    # every entry is >=2 words; TIMESTAMP_DATA is 4). Stops at the first empty word.
    entries = []  # list of (word_index, kind_name, marker_id)
    i = 0
    while i < len(words):
        word = int(words[i])
        if not (word & Profiler.ENTRY_EXISTS_BIT):
            break
        kind = (word & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
        marker_id = (word & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
        entries.append((i, _kind_name(word), marker_id))
        i += 4 if kind == EntryType.TIMESTAMP_DATA.value else 2

    # Full hex dump so the log is unambiguous even if the decode mis-strides on garbage.
    logger.info(
        "[overrun probe] math buffer window (addr=0x{:x}): {}",
        math_addr,
        " ".join(f"w{idx}=0x{int(w):08x}" for idx, w in enumerate(words)),
    )
    logger.info(
        "[overrun probe] decoded entries: {}",
        [f"w{idx}:{name}(id=0x{mid:04x})" for idx, name, mid in entries],
    )

    word0_kind = (int(words[0]) & Profiler.ENTRY_TYPE_MASK) >> Profiler.ENTRY_TYPE_SHAMT
    kernel_id = (int(words[0]) & Profiler.ENTRY_ID_MASK) >> Profiler.ENTRY_ID_SHAMT
    foreign = [(idx, name, mid) for idx, name, mid in entries if mid != kernel_id]
    healthy = word0_kind == EntryType.ZONE_START.value and not foreign

    logger.info(
        "[overrun probe] word0 kind={} kernel_id=0x{:04x} foreign_entries={} ({})",
        _kind_name(int(words[0])),
        kernel_id,
        [f"w{idx}:{name}(id=0x{mid:04x})" for idx, name, mid in foreign],
        "healthy" if healthy else "OVERRUN from unpack!",
    )

    assert healthy, (
        f"Neighbor (math) buffer corrupted: word0 kind={_kind_name(int(words[0]))} "
        f"(expected ZONE_START), foreign entries (id != KERNEL 0x{kernel_id:04x}) = "
        f"{[(idx, name, hex(mid)) for idx, name, mid in foreign]}. Unpack's deeply-nested "
        "zone closes overran write_idx into the math buffer (write-side reservation "
        "shortfall, findings sec 6.1). Bug reproduced."
    )


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
    """Phase 2b -- the overrun's production impact: it crashes a normal read.

    Same overrun kernel as test_profiler_buffer_overrun_into_neighbor, but instead of
    a raw probe we do exactly what production does: call Profiler.get_data(), which
    parses all three threads. Parsing the corrupted math buffer hits a stray ZONE_END
    with no matching ZONE_START and raises 'Possible buffer corruption' -- so the entire
    perf run fails, and the traceback blames the innocent math thread.

    Asserts the healthy behavior (a normal read succeeds and the math buffer holds only
    its own KERNEL zone), so it xfails while the bug is present and xpasses once fixed.
    """
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip()

    config = PerfConfig("sources/profiler_stress_overrun_test.cpp")
    config.generate_variant_hash()
    config.build_elfs()
    config.run_elf_files()

    # The normal production read path. With the overrun bug this raises while parsing
    # the corrupted math buffer.
    runtime = Profiler.get_data(
        config.test_name, config.variant_id, TestConfig.TENSIX_LOCATION
    )

    # If we get here, no corruption reached the parser -- the math thread should carry
    # only its own KERNEL zone, none of unpack's markers.
    math_markers = set(str(m) for m in runtime.math().raw()["marker"])
    logger.info("[overrun read] math markers = {}", math_markers)
    assert {"NEST", "FILLER"}.isdisjoint(math_markers), (
        f"math buffer contains unpack's markers {math_markers} -- overrun corruption "
        "reached the parser."
    )


@skip_for_coverage
@skip_for_quasar
def test_marker_hash_collision_detection_and_ceiling():
    """Realistic scaling limit: marker IDs are only 16 bits.

    Each profiler marker (file:line:name) is hashed to a 16-bit ID. A heavily
    instrumented kernel with many distinct zones/timestamps can therefore hit a hash
    collision (birthday bound ~300 markers). This test verifies:
      1. the hash is deterministic and stays within 16 bits,
      2. the collision-detection safety net (_assert_no_collision) fires on a real
         collision (so collisions fail loudly, never silently mis-attribute), and
      3. a collision is actually reachable, and logs at how many markers -- i.e. the
         practical ceiling on markers-per-kernel.

    Pure Python -- no hardware needed (it exercises the host-side hash/metadata code).
    """
    # 1. Deterministic + 16-bit.
    marker = "LLK_PROFILER:foo.cpp:10:MARK"
    h = Profiler._hash_meta(marker)
    assert h == Profiler._hash_meta(marker), "hash is not deterministic"
    assert 0 <= h <= 0xFFFF, f"hash {h} is not a 16-bit value"

    # 2. The safety net raises on a genuine collision (two distinct markers, same ID).
    m1 = ProfilerFullMarker(marker="A", file="f.cpp", line=1, id=0x1234)
    m2 = ProfilerFullMarker(marker="B", file="f.cpp", line=2, id=0x1234)
    with pytest.raises(AssertionError, match="collision"):
        Profiler._assert_no_collision({m1.id: m1}, m2)
    # ...and does NOT raise when the same marker is re-seen (legitimate dedup).
    Profiler._assert_no_collision({m1.id: m1}, m1)

    # 3. A 16-bit collision is reachable; find and report how many distinct markers it
    #    takes (the practical per-kernel ceiling).
    seen: dict[int, str] = {}
    collision_at = None
    for i in range(5000):
        s = f"LLK_PROFILER:gen.cpp:{i}:MARK"
        hid = Profiler._hash_meta(s)
        if hid in seen and seen[hid] != s:
            collision_at = i
            break
        seen[hid] = s

    logger.info(
        "[hash] first 16-bit marker collision after {} distinct markers", collision_at
    )
    assert (
        collision_at is not None
    ), "expected a 16-bit hash collision within 5000 markers (birthday bound ~300)"
