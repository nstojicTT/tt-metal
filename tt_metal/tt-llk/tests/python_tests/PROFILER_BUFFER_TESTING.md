# Profiler L1 Buffer — Testing & Findings (living document)

This is the running reference for the profiler buffer stress-testing work. It explains **how the
buffer works**, **what each test does and why**, **what we've proven**, and **what's left**. It is
updated as we go. If you read one file to understand this effort, read this one.

> Ultimate goal: make the profiler's `post.csv` trustworthy enough to enable **PR gating on perf
> regressions**. That requires proving we write and read the L1 buffer correctly and that overflow
> is handled predictably.

## Table of contents
1. [Big picture](#1-big-picture)
2. [The L1 buffer internals](#2-the-l1-buffer-internals)
3. [The write path (C++, on chip)](#3-the-write-path-c-on-chip)
4. [The read path (Python, host)](#4-the-read-path-python-host)
5. [Overflow: capacity math](#5-overflow-capacity-math)
6. [Known weaknesses / findings](#6-known-weaknesses--findings)
7. [The test suite](#7-the-test-suite)
8. [How to run the tests](#8-how-to-run-the-tests)
9. [Progress log](#9-progress-log)
10. [Roadmap](#10-roadmap)

---

## 1. Big picture

The profiler measures how long parts of a kernel take, in clock cycles. Each of the Tensix's
compute threads records timing "events" into a small on-chip memory buffer; the host later reads
those buffers back, decodes them, aggregates them into stats, and writes `post.csv`.

```
 (on chip, C++)          (host, Python)           (host, Python)         (not built yet)
 ┌───────────┐  read     ┌─────────────┐  stats   ┌──────────────┐  diff  ┌──────────┐
 │ L1 buffer │ ────────► │ profilerData│ ───────► │  perfData →  │ ─────► │ PR gate  │
 │ (writer)  │           │  (parser)   │          │   post.csv   │        │          │
 └───────────┘           └─────────────┘          └──────────────┘        └──────────┘
   profiler.h            profiler.py               perf.py/metrics.py
```

This document focuses on the **first two boxes** — the L1 buffer and its parser — which is where the
"stress test the buffer / write & read correctly" work lives.

---

## 2. The L1 buffer internals

### What and where
- **L1** is fast on-chip SRAM next to the compute cores. Addresses like `0x16B000` are byte offsets
  into it.
- A **Tensix** runs compute as three RISC-V cores called **TRISCs**: T0 **unpack**, T1 **math**,
  T2 **pack**. Each has its own private profiler buffer, so the three threads never write to the
  same memory.
- The three buffers are **contiguous and back-to-back** (`test_config.py`, `profiler.h`):

  | Thread | Start | End | Size |
  |--------|-------|-----|------|
  | unpack | `0x16B000` | `0x16C000` | 1024 words |
  | math   | `0x16C000` | `0x16D000` | 1024 words |
  | pack   | `0x16D000` | `0x16E000` | 1024 words |

  A "word" is 32 bits (4 bytes). Each buffer holds **`BUFFER_LENGTH = 0x400 = 1024` words**. This
  size is defined independently in C++ (`profiler.h` `BUFFER_LENGTH`) and Python
  (`test_config.py` `THREAD_PERFORMANCE_DATA_BUFFER_LENGTH`).

### Event encoding
Every event is **2 words** (or **4** for `TIMESTAMP_DATA`):

**Word 1 — the "meta word":** three fields packed into 32 bits:
```
 bit:  31    28 27              12 11           0
      ┌────────┬──────────────────┬──────────────┐
      │  KIND  │    MARKER ID      │  TIME (high) │
      │ 4 bits │     16 bits       │   12 bits    │
      └────────┴──────────────────┴──────────────┘
```
- **KIND** — event type: `TIMESTAMP`(0b1000), `TIMESTAMP_DATA`(0b1001), `ZONE_START`(0b1010),
  `ZONE_END`(0b1011). The top bit of every valid kind is 1 → it doubles as an "entry exists" flag.
- **MARKER ID** — a 16-bit hash of the marker string (see below).
- **TIME (high)** — the top 12 bits of the timestamp.

**Word 2 — `timestamp_low`:** the low 32 bits of the timestamp.

So the stored timestamp is **44 bits** (12 high + 32 low). The clock is 64-bit; the top ~20 bits are
discarded because there's no room and durations within a kernel fit easily in 44 bits.

**`TIMESTAMP_DATA`** appends **two more words** holding a 64-bit payload the kernel passes
explicitly — the only place non-timing data is stored. We use this in tests to stamp a known index
into each event.

### Entry sizes (matters for capacity)
| Kind | Words |
|------|-------|
| `TIMESTAMP` | 2 |
| `ZONE_START` / `ZONE_END` | 2 each (a zone = 4 total) |
| `TIMESTAMP_DATA` | 4 |

### Markers & metadata
A marker like `ZONE_SCOPED("MATH")` is turned into the string
`"LLK_PROFILER:<file>:<line>:MATH"`, which is hashed to the 16-bit MARKER ID (FNV-1a folded to 16
bits — `hashString16` in C++, `_hash_meta` in Python; **the two must agree**). The full string is
also stored in an ELF metadata section, which the host loads to map ID → readable
(file/line/name). That's why the parser needs a `profiler_meta` dict.

---

## 3. The write path (C++, on chip)

Source: `tests/helpers/include/profiler.h`, driven by `tests/helpers/src/trisc.cpp`.

- **`reset()`** runs once at the top of every kernel (`trisc.cpp`), `memset`-ing the buffer to 0 and
  setting `write_idx = 0`. The trailing zeros act as an end marker for the reader.
- The whole kernel is wrapped in **`ZONE_SCOPED("KERNEL")`** (`trisc.cpp`), so there is **always one
  zone open** and `write_idx` starts at 2 (after `ZONE_START(KERNEL)`).
- **`write_entry`** appends an event at `write_idx`, incrementing it by 2 (or 4).
- **`is_buffer_full()`**: `(BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4`. It gates every *new*
  event, reserving 1 word per open zone plus a 4-word margin.
- **`zone_scoped` (RAII)**: the constructor writes `ZONE_START` *only if* not full and sets
  `is_opened`; the destructor writes `ZONE_END` *only if* `is_opened`. So a zone that can't open is
  dropped **all-or-nothing** — never a start without an end.

**Overflow behavior:** when full, *new* zones/timestamps are **silently dropped**; *already-open*
zones still close. The buffer stays balanced and zero-terminated.

---

## 4. The read path (Python, host)

Source: `tests/python_tests/helpers/profiler.py`.

- **`Profiler.get_data`** reads a fixed 1024 words per thread over the debug link and parses them.
- **`_parse_thread`** walks the words: it stops at the first word without the "exists" bit (a zero),
  and for each event reads the meta word then pulls the paired `timestamp_low` (and data words for
  `TIMESTAMP_DATA`). `ZONE_END`s are matched against a stack of `ZONE_START`s.
- The result is a `ProfilerData` frame you can filter: `.unpack()/.math()/.pack()`,
  `.zones()/.timestamps()`, `.marker("X")`, then `.raw()` (parse order) or `.frame()` (paired
  zones with durations, sorted by timestamp).

---

## 5. Overflow: capacity math

For an entry of size `E` words, with the KERNEL zone open (reserving 1 word + the 4-word margin),
entries land at `write_idx = 2, 2+E, 2+2E, …` and are written while `write_idx ≤ 1019`. So:

```
capacity = number of E-word entries with (2 + k*E) ≤ 1019, k = 0,1,2,…
```

| Entry type | E | Capacity |
|------------|---|----------|
| `TIMESTAMP_DATA` | 4 | **255** (confirmed on Wormhole) |
| `ZONE_SCOPED` (START+END) | 4 | **255** (confirmed on Wormhole) |
| `TIMESTAMP` | 2 | **509** (confirmed on Wormhole) |

After the last entry, the KERNEL zone's `ZONE_END` fits in the remaining words with no overrun.
These exact numbers are used as **regression tripwires** in the tests: if the buffer size, the
guard, or the entry layout ever changes, capacity shifts and the test fails loudly.

---

## 6. Known weaknesses / findings

Found while reading the code. Details in the improvements/deep-dive notes; summarized here:

1. **Write-side reservation shortfall (overrun). ✅ CONFIRMED ON HARDWARE (Wormhole, 2026-07-22).**
   The guard reserves 1 word per open zone, but a `ZONE_END` is 2 words and the destructor writes it
   *unconditionally*. With enough zones open near a full buffer, closes push `write_idx` past the
   buffer end into the **neighbor thread's** buffer — cross-thread L1 corruption. Reproduced by
   `test_profiler_buffer_overrun_into_neighbor`: after unpack filled near-full and nested ~6 zones,
   the math buffer's first word (which should be math's own `KERNEL ZONE_START`) came back as a stray
   `ZONE_END` (`word0 = 0xb7c537f7`, KIND = `0b1011`). Safe in *today's* kernels only because they
   nest shallowly (depth 1–2). **Fix:** reserve 2 words per open zone in `is_buffer_full`, or check
   capacity on the close path. (Fixing this is LLK-core C++ — needs mentor sign-off + metal-integration
   review; test is currently `xfail`.)
2. **Reader crashes opaquely on a malformed buffer.** `_parse_thread` reads the paired word with an
   unguarded `next()`; on a truncated buffer it raises a bare `StopIteration` with no diagnostic.
   Happens only on a malformed buffer (overrun, length drift, early read, format change), not in
   normal runs. → Phase 3 fixes + tests this.
3. **Overflow drops data silently.** No warning/flag when events are dropped → a truncated run can
   feed wrong-but-plausible numbers into a gate. → open contract question for the mentor.
4. **Profiler is hard to unit-test in isolation.** `profiler.py` pulls in torch/ttexalens/etc. on
   import → pure-Python tests need an isolated harness or light decoupling. → gates Phase 3.

---

## 7. The test suite

Files: `tests/sources/profiler_stress_*.cpp` (kernels) and
`tests/python_tests/test_profiler_stress.py` (tests). Every test uses the same idea: **flood a
buffer past capacity with a verifiable pattern, then check the write→L1→read round trip.**

### Phase 1 / 1b tests

| Test | Kernel | What it floods | Checks | Expected |
|------|--------|----------------|--------|----------|
| `test_profiler_buffer_overflow_stress` | `profiler_stress_test.cpp` | `TIMESTAMP_DATA` on **all 3 threads** | each thread's data is a clean contiguous prefix `0..K-1`, exact capacity | 255 / thread |
| `test_profiler_buffer_overflow_zones` | `profiler_stress_zones_test.cpp` | `ZONE_SCOPED` on unpack | all retained zones are paired (have a duration), exact capacity | 255 |
| `test_profiler_buffer_overflow_timestamps` | `profiler_stress_timestamps_test.cpp` | `TIMESTAMP` (2-word) on unpack | timestamps strictly increasing, exact (larger) capacity | 509 |

**Why these three:**
- **TIMESTAMP_DATA, all threads** — the payload lets us verify *fidelity* (write==read) and *order*,
  and covers all three buffers (not just unpack) + confirms no cross-thread bleed.
- **ZONE_SCOPED** — exercises the *pairing* logic and proves overflow drops *whole* zones (the
  all-or-nothing behavior), never a half-open zone.
- **TIMESTAMP (2-word)** — exercises the 2-word entry path and a *different* capacity, so the guard
  is tested at a second boundary. Monotonic timestamps also sanity-check the 44-bit decode.

**What they prove together:** the buffer writes and reads correctly across all entry sizes and all
three threads, overflow degrades cleanly (tail-drop, balanced, no corruption), and capacity matches
theory exactly — all guarded by exact tripwires.

**What they do NOT cover yet:** malformed-buffer read robustness (Phase 3).

### Phase 2 test (the overrun hunt) — ✅ BUG CAUGHT

| Test | Kernel | What it does | Detection |
|------|--------|--------------|-----------|
| `test_profiler_buffer_overrun_into_neighbor` | `profiler_stress_overrun_test.cpp` | unpack fills near-full, then opens a deep nest of zones so the closes overrun into the math buffer | reads the math buffer's first word directly; healthy = its own `ZONE_START`, corrupted = a stray `ZONE_END` from unpack |

Unlike Phase 1/1b (which assert *correct* behavior), this test asserts the neighbor is *healthy* —
so **a failure means we reproduced the overrun bug** (findings §6.1). It is confirmed and marked
`xfail`. Full walkthrough below.

#### The bug, in code (`tests/helpers/include/profiler.h`)
- `is_buffer_full()` (`profiler.h:131`): `return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;`
  — reserves **1 word per open zone** plus a 4-word margin, and gates every *new* write.
- `write_entry()` (`profiler.h:140`): writes **2 words** per entry (meta word + `timestamp_low`).
- `zone_scoped` destructor (`profiler.h:182`): `if (is_opened) { write_entry(ZONE_END, id16); … }`
  — writes the 2-word `ZONE_END` **unconditionally**, with *no* `is_buffer_full` check.
- **The mismatch:** the guard reserves 1 word/open-zone, but each close costs **2** → short by
  `open_zone_cnt` words. The unconditional close is where the cursor runs off the end.
- The three buffers are contiguous (`test_config.py:207`: unpack `0x16B000` → math `0x16C000`), so
  an unpack overrun spills into the **start of the math buffer**.

#### The reproduction (`sources/profiler_stress_overrun_test.cpp`)
1. Emit **501** flat `TIMESTAMP("FILLER")` (2 words each) → `write_idx = 2` (after
   `ZONE_SCOPED("KERNEL")` in `trisc.cpp:91`) `+ 2×501 = 1004`.
2. `open_nested_zones(20)` recurses, opening one `ZONE_SCOPED("NEST")` per frame. The guard lets
   ~6 open before blocking (each open raises `open_zone_cnt`, tightening the guard); deeper frames
   get `is_opened == false` and write nothing.
3. On unwind, the ~6 open `NEST` zones **plus** the enclosing `KERNEL` zone all close — 7
   unconditional 2-word `ZONE_END`s = 14 words — driving `write_idx` from ~1016 to ~1030. Indices
   1024–1029 land **outside** the 1024-word buffer, in math's `buffer[0..5]`.

#### The detection (`test_profiler_buffer_overrun_into_neighbor`)
- Math emits nothing, so its buffer holds only its own `KERNEL` zone; its **first word must be a
  `ZONE_START`**.
- We read math's raw words with `read_words_from_device` (bypassing the parser, which would raise
  the "buffer corruption" error at `profiler.py:485`), and inspect `word[0]`'s KIND field.
- Healthy → `ZONE_START` (`0b1010`); overrun → `ZONE_END` (`0b1011`), a stray close from unpack.
- Unpack does 500+ events and finishes long after math (which is empty), so the overrun reliably
  overwrites math's already-written first word — making the detection deterministic.

#### The evidence
```
[overrun probe] math word0=0xb7c537f7 kind=ZONE_END (OVERRUN from unpack!)
```
`0xb7c537f7` → KIND `0xb` = `ZONE_END`, marker_id `0x7c53` (a `NEST` close), `time_high 0x7f7`.
Reproduced across runs deterministically — the marker_id and KIND are identical each time; only the
timestamp bits vary (e.g. a later run gave `0xb7c5384f`). Only unpack overrunning its buffer
boundary could place a `ZONE_END` in math's first word.

#### Disposition
Test marked `xfail` (strict=False): documents the confirmed bug, keeps the suite green, and will
`xpass` when the reservation is fixed (then remove the marker). **Fix:** reserve 2 words per open
zone in `is_buffer_full`, or check capacity on the close path — LLK-core C++, pending mentor
sign-off + metal-integration review.

### Further hunts (#1 overrun→crash, #3 hash ceiling)

| Test | Realistic trigger | What it shows |
|------|-------------------|----------------|
| `test_profiler_overrun_crashes_normal_read` | a deep-nesting kernel near full (same as Phase 2) | escalates §6.1: the overrun-corrupted neighbor makes a **normal** `Profiler.get_data()` raise `"Possible buffer corruption"` — the whole perf run dies, blaming the innocent math thread. Asserts a normal read succeeds → `xfail` while the bug is present. |
| `test_marker_hash_collision_detection_and_ceiling` | a heavily-instrumented kernel with many distinct zones/timestamps | marker IDs are only **16 bits** (`_hash_meta`), so ~300 markers → birthday collision. Verifies the hash is 16-bit + deterministic, the `_assert_no_collision` safety net fires on a real collision (fails loud, never silently mis-attributes), and reports how many distinct markers it takes to collide (the practical per-kernel ceiling). **Pure Python — no hardware.** |

`test_profiler_overrun_crashes_normal_read` is the *production* face of the Phase 2 bug: Phase 2
proved corruption with a raw probe; this shows a normal read crashes. The hash-ceiling test is the
first pure-logic test here — it foreshadows Phase 3's placement question (it doesn't need a device
but currently rides the hardware conftest).

---

## 8. How to run the tests

On a machine with a Tenstorrent card:

```bash
cd tt_metal/tt-llk/tests
./setup_testing_env.sh          # once — installs the SFPI compiler
pip install -r requirements.txt  # once
export CHIP_ARCH=wormhole        # or blackhole, matching the card
cd python_tests
pytest test_profiler_stress.py   # compiles + runs + reads + checks; output to terminal
```

- **Output** goes to the terminal: a `[stress ...] emitted=… retained=… dropped=…` log line per
  test (pytest.ini has `log_cli = true`), then PASS/FAIL.
- **No separate compile step** — the test compiles the kernel itself via `build_elfs()`.
- **⚠️ Build cache gotcha:** the build is cached by a config hash, **not** by source content
  (`.build_complete` marker under `/tmp/tt-llk-build/`). If you edit a `.cpp`, the cache won't
  notice and you'll run the **stale** binary. After any `.cpp` change:
  ```bash
  rm -rf /tmp/tt-llk-build/
  pytest test_profiler_stress.py
  ```
  (Pure-Python edits need no rebuild.)

---

## 9. Progress log

- **2026-07-22 — Phase 1 (TIMESTAMP_DATA overflow, unpack): PASS on Wormhole.**
  `emitted=600 retained=255 dropped=345`. Capacity 255 matched theory exactly. Locked in as an exact
  regression assertion. Proved: end-to-end round trip on silicon, write==read fidelity, clean
  tail-drop, exact capacity, no parser crash on a full buffer.
- **2026-07-22 — Phase 1b: PASS on Wormhole (3 tests).**
  All predicted capacities confirmed exactly: TIMESTAMP_DATA on unpack/math/pack = 255 each;
  ZONE_SCOPED = 255; 2-word TIMESTAMP = 509. Proved: all three buffers write/read correctly and
  independently (no cross-thread bleed); zone START/END pairing survives a full buffer with whole-zone
  drops; the 2-word entry path parses correctly at its own (larger) capacity; and the capacity model
  holds across entry sizes. All locked in as exact regression tripwires.
- **2026-07-22 — Phase 2: BUG CAUGHT on Wormhole.**
  `test_profiler_buffer_overrun_into_neighbor` reproduced the write-side reservation overrun (§6.1).
  Unpack filled near-full (501 fillers) then nested ~6 zones; closing them overran `write_idx` past
  its 1024-word buffer into the math buffer, overwriting math's first word with a stray `ZONE_END`
  (`word0 = 0xb7c537f7`). The three capacity tests stayed green. Test marked `xfail` to document the
  bug; fix (reserve 2 words/zone or check on close) pending mentor sign-off.
- **2026-07-22 — Added hunts #1 and #3 [awaiting hardware run for #1].**
  `test_profiler_overrun_crashes_normal_read` (#1) — the overrun's production face: a normal
  `get_data()` should raise on the corrupted neighbor (`xfail`). `test_marker_hash_collision_detection_and_ceiling`
  (#3) — verifies the 16-bit marker-hash safety net and quantifies the collision ceiling (pure Python).

---

## 10. Roadmap

- **Phase 2 — provoke the write-side overrun. [IMPLEMENTED, awaiting hardware run]** See the Phase 2
  test above. `rm -rf /tmp/tt-llk-build/` then run; read the `[overrun probe]` log line. If the math
  first word comes back as `ZONE_END`, we caught the bug.
- **Phase 3 — read-side robustness (pure-Python).** Guard the parser's `next()` so a truncated
  buffer raises a clear error instead of `StopIteration`; unit-test with synthetic malformed
  buffers. Blocked on the test-placement/decoupling decision.
- **Open mentor decisions:** silent-drop contract; whether to fix the reservation bug (informed by
  Phase 2); where pure unit tests live.
