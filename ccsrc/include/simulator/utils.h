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

#ifndef INCLUDE_QUANTUMSTATE_UTILS_HPP
#define INCLUDE_QUANTUMSTATE_UTILS_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef USE_OPENMP
#    include <omp.h>
#endif

#include "core/mq_base_types.h"

namespace mindquantum::sim {
index_t QIndexToMask(qbits_t objs);
PauliMask GenPauliMask(const std::vector<PauliWord>& pws);
struct SingleQubitGateMask {
    qbit_t q0 = 0;
    qbits_t ctrl_qubits{};
    index_t obj_mask = static_cast<uint64_t>(0);
    index_t ctrl_mask = static_cast<uint64_t>(0);
    index_t obj_high_mask = static_cast<uint64_t>(0);
    index_t obj_low_mask = static_cast<uint64_t>(0);

    SingleQubitGateMask(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
};
struct DoubleQubitGateMask {
    qbit_t q_min = 0;
    qbit_t q_max = 0;
    qbits_t ctrl_qubits{};
    index_t obj_min_mask = static_cast<uint64_t>(0);
    index_t obj_max_mask = static_cast<uint64_t>(0);
    index_t obj_mask = static_cast<uint64_t>(0);
    index_t ctrl_mask = static_cast<uint64_t>(0);
    index_t obj_high_mask = static_cast<uint64_t>(0);
    index_t obj_rev_high_mask = static_cast<uint64_t>(0);
    index_t obj_low_mask = static_cast<uint64_t>(0);
    index_t obj_rev_low_mask = static_cast<uint64_t>(0);

    DoubleQubitGateMask(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
};

#define SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, ori, des)                      \
    do {                                                                                                               \
        (des) = (((ori) & (obj_rev_low_mask)) << 1) + ((ori) & (obj_low_mask));                                        \
        (des) = (((des) & (obj_rev_high_mask)) << 1) + ((des) & (obj_high_mask));                                      \
    } while (0)

// Number of independent worker threads for a batch of gradient evaluations
// over a state of `n_amplitudes` amplitudes.  Full-state kernels already
// consume an OpenMP team above `parallel_threshold`, so serializing the outer
// batch there avoids nested teams and memory-bandwidth oversubscription; small
// states retain independent batch workers.
inline auto SelectGradientBatchThreads(index_t n_amplitudes, index_t parallel_threshold, size_t requested_threads)
    -> size_t {
    if (requested_threads == 0) {
        throw std::invalid_argument("Gradient batch thread count cannot be zero.");
    }
#ifdef USE_OPENMP
    if (n_amplitudes >= parallel_threshold && omp_get_max_threads() > 1) {
        return 1;
    }
#else
    (void) n_amplitudes;
    (void) parallel_threshold;
#endif
    return requested_threads;
}

// Runs `task(index)` for every index in [0, count) on at most `threads`
// worker threads, on the calling thread when one suffices.  The first
// exception thrown by a task cancels the remaining work and is rethrown on
// the calling thread once every worker has joined.
template <typename task_t>
void ParallelBatch(size_t threads, size_t count, const task_t& task) {
    if (count == 0) {
        throw std::invalid_argument("Gradient parameter batch cannot be empty.");
    }
    if (threads == 0) {
        throw std::invalid_argument("Gradient batch thread count cannot be zero.");
    }
    threads = std::min(threads, count);
    if (threads == 1) {
        for (size_t index = 0; index < count; ++index) {
            task(index);
        }
        return;
    }

    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::atomic<bool> failed{false};
    const auto run_chunk = [&](size_t start, size_t end) {
        for (size_t index = start; index < end && !failed.load(std::memory_order_relaxed); ++index) {
            try {
                task(index);
            } catch (...) {
                const std::lock_guard<std::mutex> lock(failure_mutex);
                if (failure == nullptr) {
                    failure = std::current_exception();
                }
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(threads);
    const auto chunk = count / threads;
    const auto remainder = count % threads;
    size_t end = 0;
    for (size_t worker = 0; worker < threads; ++worker) {
        const auto start = end;
        end = start + chunk + (worker < remainder ? 1 : 0);
        workers.emplace_back(run_chunk, start, end);
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}
}  // namespace mindquantum::sim

#endif
