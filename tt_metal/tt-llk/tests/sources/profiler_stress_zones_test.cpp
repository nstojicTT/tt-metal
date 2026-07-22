// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "profiler.h"

struct RuntimeParams
{
};

constexpr std::uint32_t STRESS_EVENT_COUNT = 600;

#ifdef LLK_TRISC_UNPACK

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
    for (std::uint32_t i = 0; i < STRESS_EVENT_COUNT; i++)
    {
        ZONE_SCOPED("STRESS_ZONE");
    }
}

#endif

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
