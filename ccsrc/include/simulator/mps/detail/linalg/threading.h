/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_THREADING_HPP
#define INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_THREADING_HPP

#include <stdexcept>

#include "simulator/mps/detail/linalg/symbol.h"

namespace mindquantum::sim::mps::detail::linalg {
extern "C" {
int MQ_LINALG_SYM(openblas_get_num_threads)();
void MQ_LINALG_SYM(openblas_set_num_threads)(int);
}  // extern "C"

// Pins the OpenBLAS thread count for the lifetime of the object and restores the previous value afterwards. Used around
// regions that already parallelise over independent BLAS calls, where OpenBLAS-internal threading would oversubscribe.
class ScopedOpenBlasNumThreads {
 public:
    explicit ScopedOpenBlasNumThreads(int num_threads) : prev_threads_(MQ_LINALG_SYM(openblas_get_num_threads)()) {
        if (num_threads <= 0) {
            throw std::invalid_argument("mqmps: OpenBLAS thread count must be positive.");
        }
        if (prev_threads_ != num_threads) {
            MQ_LINALG_SYM(openblas_set_num_threads)(num_threads);
        }
    }

    ~ScopedOpenBlasNumThreads() {
        MQ_LINALG_SYM(openblas_set_num_threads)(prev_threads_);
    }

    ScopedOpenBlasNumThreads(const ScopedOpenBlasNumThreads&) = delete;
    ScopedOpenBlasNumThreads& operator=(const ScopedOpenBlasNumThreads&) = delete;
    ScopedOpenBlasNumThreads(ScopedOpenBlasNumThreads&&) = delete;
    ScopedOpenBlasNumThreads& operator=(ScopedOpenBlasNumThreads&&) = delete;

 private:
    int prev_threads_;
};
}  // namespace mindquantum::sim::mps::detail::linalg

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_THREADING_HPP
