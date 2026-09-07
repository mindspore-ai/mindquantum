/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MQ_CONFIG_OPENMP_HPP
#define MQ_CONFIG_OPENMP_HPP

#include <algorithm>
#include <cstdint>
#include <thread>

#ifdef USE_OPENMP
#    include <omp.h>
#endif

#include "config/details/macros.h"

// How MindQuantum sizes its OpenMP teams.
//
// A team is sized from the work it has to do, not from the number of cores:
// one thread per kMinWorkPerThread work items (amplitudes touched), capped by
// the OpenMP thread limit.  A 14-qubit state vector therefore runs on two
// threads whether the host has 8 cores or 128, and a 30-qubit state uses every
// thread the user allows.  The cap is the standard OpenMP one, OMP_NUM_THREADS
// (default: all cores), so users bound the parallelism with the knob they
// already know.  Teams are never nested.
namespace omp {
#ifdef _MSC_VER
using idx_t = int64_t;
#else
using idx_t = uint64_t;
#endif  // _MSC_VER

// Work a thread must own before a region amortizes its fork/join and barrier
// cost.  A region with less than twice this work runs on the calling thread.
inline constexpr uint64_t kMinWorkPerThread = uint64_t{1} << 13;

// Upper bound on the threads MindQuantum runs at once: the OpenMP thread
// limit, or the hardware concurrency when built without OpenMP.
inline int MaxThreads() noexcept {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
#endif
}

// Threads a region over `work` uniform items should run on.  Returns 1,
// meaning "run on the calling thread", for small work, inside an active
// OpenMP team and in builds without OpenMP.  Small work is decided without
// touching the OpenMP runtime, which registers every foreign thread that
// calls into it; batch workers on small states therefore stay plain threads.
inline int TeamSize(uint64_t work) noexcept {
#ifdef USE_OPENMP
    const auto wanted = work / kMinWorkPerThread;
    if (wanted < 2 || omp_in_parallel()) {
        return 1;
    }
    return static_cast<int>(std::min(wanted, static_cast<uint64_t>(MaxThreads())));
#else
    (void) work;
    return 1;
#endif
}

// Threads for `count` independent items that each cost `work_per_item`, e.g.
// Hamiltonian terms over a state of `work_per_item` amplitudes.  Never more
// than one thread per item.
inline int TeamSize(uint64_t count, uint64_t work_per_item) noexcept {
    const auto team = static_cast<uint64_t>(TeamSize(count * work_per_item));
    return static_cast<int>(std::clamp(count, uint64_t{1}, team));
}
}  // namespace omp

// Runs the loop statement `...` under `omp parallel for <clauses>` on a team
// of `team` threads, or plainly on the calling thread when `team` is 1.
// `clauses` are appended to the directive verbatim, e.g.
// `schedule(static) reduction(+ : real, imag)`.
#define MQ_OMP_PARALLEL_FOR_TEAM(team, clauses, ...)                                                                   \
    if (const int mq_omp_team_ = (team); mq_omp_team_ == 1) {                                                          \
        __VA_ARGS__                                                                                                    \
    } else {                                                                                                           \
        MQ_DO_PRAGMA(omp parallel for num_threads(mq_omp_team_) clauses)                                               \
        __VA_ARGS__                                                                                                    \
    }

// MQ_OMP_PARALLEL_FOR_TEAM on a team sized for `work` uniform items.
#define MQ_OMP_PARALLEL_FOR_CLAUSES(work, clauses, ...)                                                                \
    MQ_OMP_PARALLEL_FOR_TEAM(::omp::TeamSize(work), clauses, __VA_ARGS__)

// MQ_OMP_PARALLEL_FOR_CLAUSES with a static schedule and no further clauses.
#define MQ_OMP_PARALLEL_FOR(work, ...) MQ_OMP_PARALLEL_FOR_CLAUSES(work, schedule(static), __VA_ARGS__)

#endif /* MQ_CONFIG_OPENMP_HPP */
