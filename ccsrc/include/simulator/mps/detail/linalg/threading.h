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

#if !defined(BLAS_SYMBOL_PREFIX)
#define MQ_LINALG_THREADING_DEFINED_BLAS_SYMBOL_PREFIX
#define BLAS_SYMBOL_PREFIX
#endif

#define MQ_LINALG_THREADING_CAT_IMPL(x, y) x##y
#define MQ_LINALG_THREADING_CAT(x, y) MQ_LINALG_THREADING_CAT_IMPL(x, y)
#define MQ_LINALG_THREADING_SYM(name) MQ_LINALG_THREADING_CAT(BLAS_SYMBOL_PREFIX, name)

namespace mindquantum::sim::mps::detail::linalg {
namespace threading::detail {
struct OpenBlasApi {
    using get_fn = int (*)();
    using set_fn = void (*)(int);

    get_fn get;
    set_fn set;
};

extern "C" {
int MQ_LINALG_THREADING_SYM(openblas_get_num_threads)();
void MQ_LINALG_THREADING_SYM(openblas_set_num_threads)(int);
}  // extern "C"

inline OpenBlasApi LoadOpenBlasApi() noexcept {
    return OpenBlasApi{
        &MQ_LINALG_THREADING_SYM(openblas_get_num_threads),
        &MQ_LINALG_THREADING_SYM(openblas_set_num_threads),
    };
}

inline const OpenBlasApi& OpenBlas() noexcept {
    static const OpenBlasApi api = LoadOpenBlasApi();
    return api;
}
}  // namespace threading::detail

class ScopedOpenBlasNumThreads {
 public:
    explicit ScopedOpenBlasNumThreads(int num_threads) {
        if (num_threads <= 0) {
            throw std::invalid_argument("mqmps: OpenBLAS thread count must be positive.");
        }
        const auto& api = threading::detail::OpenBlas();
        prev_threads_ = api.get();
        restore_ = true;
        if (prev_threads_ != num_threads) {
            api.set(num_threads);
        }
    }

    ~ScopedOpenBlasNumThreads() {
        if (!restore_) {
            return;
        }
        const auto& api = threading::detail::OpenBlas();
        api.set(prev_threads_);
    }

    ScopedOpenBlasNumThreads(const ScopedOpenBlasNumThreads&) = delete;
    ScopedOpenBlasNumThreads& operator=(const ScopedOpenBlasNumThreads&) = delete;
    ScopedOpenBlasNumThreads(ScopedOpenBlasNumThreads&&) = delete;
    ScopedOpenBlasNumThreads& operator=(ScopedOpenBlasNumThreads&&) = delete;

 private:
    int prev_threads_ = 1;
    bool restore_ = false;
};
}  // namespace mindquantum::sim::mps::detail::linalg

#undef MQ_LINALG_THREADING_SYM
#undef MQ_LINALG_THREADING_CAT
#undef MQ_LINALG_THREADING_CAT_IMPL
#ifdef MQ_LINALG_THREADING_DEFINED_BLAS_SYMBOL_PREFIX
#undef MQ_LINALG_THREADING_DEFINED_BLAS_SYMBOL_PREFIX
#undef BLAS_SYMBOL_PREFIX
#endif

#endif  // INCLUDE_SIMULATOR_MPS_DETAIL_LINALG_THREADING_HPP
