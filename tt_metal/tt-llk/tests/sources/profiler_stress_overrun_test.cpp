// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "profiler.h"

struct RuntimeParams
{
};

// Phase 2 — provoke the write-side reservation overrun (see PROFILER_BUFFER_TESTING.md sec 6.1).
//
// is_buffer_full() reserves only 1 word per open zone, but each ZONE_END is 2 words and the
// zone destructor writes it *unconditionally* (no capacity check). So if several zones are open
// near a full buffer, closing them pushes write_idx past the 1024-word buffer into the adjacent
// MATH buffer (the three thread buffers are contiguous in L1).
//
// Recipe: fill to near-full with flat 2-word timestamps, then open a deep nest of zones.
//   501 fillers -> write_idx = 2 (KERNEL start) + 2*501 = 1004.
//   The guard then lets ~6 nested zones open before blocking further opens; unwinding their
//   closes (plus the enclosing KERNEL zone) drives write_idx to ~1030 -> ~6 words spill into
//   the math buffer. The overrun width (~6 words) leaves margin so it triggers robustly rather
//   than at a single knife-edge fill level.
constexpr std::uint32_t FILLER_COUNT = 501;
constexpr std::uint32_t NEST_DEPTH   = 20; // the guard caps how many actually open (~6)

#ifdef LLK_TRISC_UNPACK

// Open `depth` nested zones. Each zone stays open across the recursive call and closes on unwind,
// so at the deepest point all opened zones are live simultaneously. The ZONE_END on unwind is the
// unconditional write that overruns.
static void open_nested_zones(std::uint32_t depth)
{
    if (depth == 0)
    {
        return;
    }
    ZONE_SCOPED("NEST");
    open_nested_zones(depth - 1);
}

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
    for (std::uint32_t i = 0; i < FILLER_COUNT; i++)
    {
        TIMESTAMP("FILLER");
    }
    open_nested_zones(NEST_DEPTH);
}

#endif

// Math and pack stay empty on purpose: the math buffer must contain only its own KERNEL zone, so
// any stray entry there is unambiguous evidence that unpack overran into it.

#ifdef LLK_TRISC_MATH

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
}

#endif
