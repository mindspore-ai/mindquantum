//   Copyright 2023 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "math/tensor/ops_cpu/memory_operator.hpp"

#include <stdexcept>

#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
Tensor init(size_t len, TDtype dtype) {
    switch (dtype) {
        case TDtype::Float32: {
            return init<TDtype::Float32>(len);
        }
        case TDtype::Float64: {
            return init<TDtype::Float64>(len);
        }
        case TDtype::Complex64: {
            return init<TDtype::Complex64>(len);
        }
        case TDtype::Complex128: {
            return init<TDtype::Complex128>(len);
        }
        default:
            throw std::runtime_error("init not implement for type");
    }
}

// -----------------------------------------------------------------------------
#define TENSOR_CAST_TO_BRANCH(src_t)                                                                                   \
    case (src_t): {                                                                                                    \
        switch (des) {                                                                                                 \
            case (TDtype::Float32):                                                                                    \
                return cast_to<src_t, TDtype::Float32>(data, len);                                                     \
            case (TDtype::Float64):                                                                                    \
                return cast_to<src_t, TDtype::Float64>(data, len);                                                     \
            case (TDtype::Complex64):                                                                                  \
                return cast_to<src_t, TDtype::Complex64>(data, len);                                                   \
            case (TDtype::Complex128):                                                                                 \
                return cast_to<src_t, TDtype::Complex128>(data, len);                                                  \
        }                                                                                                              \
    }
Tensor cast_to(const Tensor& t, TDtype des) {
    auto& data = t.data;
    const auto& src = t.dtype;
    const auto& len = t.dim;
    switch (src) {
        TENSOR_CAST_TO_BRANCH(TDtype::Float32)
        TENSOR_CAST_TO_BRANCH(TDtype::Float64)
        TENSOR_CAST_TO_BRANCH(TDtype::Complex64)
        TENSOR_CAST_TO_BRANCH(TDtype::Complex128)
    }
}
#undef TENSOR_CAST_TO_BRANCH
// -----------------------------------------------------------------------------

std::string to_string(const Tensor& t, bool simplify) {
    switch (t.dtype) {
        case (TDtype::Float32):
            return to_string<TDtype::Float32>(t.data, t.dim, simplify);
        case (TDtype::Float64):
            return to_string<TDtype::Float64>(t.data, t.dim, simplify);
        case (TDtype::Complex64):
            return to_string<TDtype::Complex64>(t.data, t.dim, simplify);
        case (TDtype::Complex128):
            return to_string<TDtype::Complex128>(t.data, t.dim, simplify);
        default:
            throw std::runtime_error("cannot convert " + to_string(t.dtype) + " to string.");
    }
    return "";
}

// -----------------------------------------------------------------------------
void destroy(Tensor* t) {
    if (t->data != nullptr) {
        free(t->data);
        t->data = nullptr;
        t->dim = 0;
    }
}

Tensor copy(const Tensor& t) {
    switch (t.dtype) {
        case TDtype::Float32:
            return cpu::copy<TDtype::Float32>(t.data, t.dim);
        case TDtype::Float64:
            return cpu::copy<TDtype::Float64>(t.data, t.dim);
        case TDtype::Complex64:
            return cpu::copy<TDtype::Complex64>(t.data, t.dim);
        case TDtype::Complex128:
            return cpu::copy<TDtype::Complex128>(t.data, t.dim);
    }
}

void* copy_mem(void* data, TDtype dtype, size_t len) {
    switch (dtype) {
        case (TDtype::Float32):
            return copy_mem<TDtype::Float32>(data, len);
        case (TDtype::Float64):
            return copy_mem<TDtype::Float64>(data, len);
        case (TDtype::Complex64):
            return copy_mem<TDtype::Complex64>(data, len);
        case (TDtype::Complex128):
            return copy_mem<TDtype::Complex128>(data, len);
    }
}

// -----------------------------------------------------------------------------
#define SET_TENSOR_BY_TENSOR(src_dtype)                                                                                \
    case (src_dtype): {                                                                                                \
        switch (source.dtype) {                                                                                        \
            case (TDtype::Float32):                                                                                    \
                set<to_device_t<src_dtype>, to_device_t<TDtype::Float32>>(t->data, source.data, t->dim, idx);          \
                break;                                                                                                 \
            case (TDtype::Float64):                                                                                    \
                set<to_device_t<src_dtype>, to_device_t<TDtype::Float64>>(t->data, source.data, t->dim, idx);          \
                break;                                                                                                 \
            case (TDtype::Complex64):                                                                                  \
                set<to_device_t<src_dtype>, to_device_t<TDtype::Complex64>>(t->data, source.data, t->dim, idx);        \
                break;                                                                                                 \
            case (TDtype::Complex128):                                                                                 \
                set<to_device_t<src_dtype>, to_device_t<TDtype::Complex128>>(t->data, source.data, t->dim, idx);       \
                break;                                                                                                 \
        }                                                                                                              \
        break;                                                                                                         \
    }
void set(Tensor* t, const Tensor& source, size_t idx) {
    if (source.dim != 1) {
        throw std::runtime_error("For set method, source tensor should have only one element.");
    }
    switch (t->dtype) {
        SET_TENSOR_BY_TENSOR(TDtype::Float32)
        SET_TENSOR_BY_TENSOR(TDtype::Float64)
        SET_TENSOR_BY_TENSOR(TDtype::Complex64)
        SET_TENSOR_BY_TENSOR(TDtype::Complex128)
    }
}
#undef SET_TENSOR_BY_TENSOR

// -----------------------------------------------------------------------------

Tensor get(const Tensor& t, size_t idx) {
    if (idx >= t.dim) {
        std::cout << t << std::endl;
        std::cout << idx << std::endl;
        throw std::runtime_error("index out of range for get.");
    }
    auto out = cpu::init(1, t.dtype);
    std::memcpy(out.data, reinterpret_cast<uint8_t*>(t.data) + idx * bit_size(t.dtype), bit_size(t.dtype));
    return out;
}
}  // namespace tensor::ops::cpu
