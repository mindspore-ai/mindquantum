/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_WORKER_SCRATCH_H_
#define INCLUDE_SIMULATOR_WORKER_SCRATCH_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "config/openmp.h"

namespace mindquantum::sim {
template <typename T>
class WorkerScratch {
    static_assert(std::is_trivial_v<T>);

 public:
    void Reserve(size_t elements, size_t workers, size_t alignment) {
        if (elements == 0) {
            return;
        }
        const auto padding = (alignment - elements % alignment) % alignment;
        constexpr auto max_count = std::numeric_limits<size_t>::max() / sizeof(T);
        if (elements > max_count - padding || elements + padding > (max_count - (alignment - 1)) / workers) {
            throw std::bad_array_new_length();
        }
        const auto stride = elements + padding;
        const auto count = workers * stride + alignment - 1;
        if (count > capacity_) {
            storage_.reset(new T[count]);
            capacity_ = count;
        }
        const auto misalignment = reinterpret_cast<uintptr_t>(storage_.get()) % (alignment * sizeof(T));
        first_ = storage_.get() + (misalignment == 0 ? 0 : alignment - misalignment / sizeof(T));
        stride_ = stride;
        workers_ = workers;
    }

    [[nodiscard]] auto ForWorker(size_t worker) const noexcept -> T* {
        assert(worker < workers_);
        return first_ + worker * stride_;
    }

    [[nodiscard]] static auto CurrentWorker() noexcept -> size_t {
#ifdef USE_OPENMP
        return static_cast<size_t>(omp_get_thread_num());
#else
        return 0;
#endif
    }

 private:
    std::unique_ptr<T[]> storage_;
    T* first_ = nullptr;
    size_t capacity_ = 0;
    size_t stride_ = 0;
    size_t workers_ = 0;
};
}  // namespace mindquantum::sim

#endif  // INCLUDE_SIMULATOR_WORKER_SCRATCH_H_
