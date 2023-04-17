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

#include "math/tensor/ops_cpu/advance_math.hpp"

#include <cstdint>
#include <stdexcept>

#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
Tensor real(const Tensor& t) {
    switch (t.dtype) {
        case TDtype::Float32:
            return cpu::real<TDtype::Float32>(t.data, t.dim);
        case TDtype::Float64:
            return cpu::real<TDtype::Float64>(t.data, t.dim);
        case TDtype::Complex128:
            return cpu::real<TDtype::Complex128>(t.data, t.dim);
        case TDtype::Complex64:
            return cpu::real<TDtype::Complex64>(t.data, t.dim);
    }
}
Tensor imag(const Tensor& t) {
    switch (t.dtype) {
        case TDtype::Float32:
            return cpu::imag<TDtype::Float32>(t.data, t.dim);
        case TDtype::Float64:
            return cpu::imag<TDtype::Float64>(t.data, t.dim);
        case TDtype::Complex128:
            return cpu::imag<TDtype::Complex128>(t.data, t.dim);
        case TDtype::Complex64:
            return cpu::imag<TDtype::Complex64>(t.data, t.dim);
    }
}

// -----------------------------------------------------------------------------

Tensor conj(const Tensor& t) {
    switch (t.dtype) {
        case TDtype::Float32:
            return ops::cpu::conj<TDtype::Float32>(t.data, t.dim);
        case TDtype::Float64:
            return ops::cpu::conj<TDtype::Float64>(t.data, t.dim);
        case TDtype::Complex64:
            return ops::cpu::conj<TDtype::Complex64>(t.data, t.dim);
        case TDtype::Complex128:
            return ops::cpu::conj<TDtype::Complex128>(t.data, t.dim);
    }
}

// -----------------------------------------------------------------------------

#define VDOT_TENSOR(bra_t)                                                                                             \
    case bra_t: {                                                                                                      \
        switch (ket_t) {                                                                                               \
            case TDtype::Float32: {                                                                                    \
                return ops::cpu::vdot<bra_t, TDtype::Float32>(bra.data, bra.dim, ket.data);                            \
            }                                                                                                          \
            case TDtype::Float64: {                                                                                    \
                return ops::cpu::vdot<bra_t, TDtype::Float64>(bra.data, bra.dim, ket.data);                            \
            }                                                                                                          \
            case TDtype::Complex64: {                                                                                  \
                return ops::cpu::vdot<bra_t, TDtype::Complex64>(bra.data, bra.dim, ket.data);                          \
            }                                                                                                          \
            case TDtype::Complex128: {                                                                                 \
                return ops::cpu::vdot<bra_t, TDtype::Complex128>(bra.data, bra.dim, ket.data);                         \
            }                                                                                                          \
        }                                                                                                              \
        break;                                                                                                         \
    }
Tensor vdot(const Tensor& bra, const Tensor& ket) {
    if (bra.dim != ket.dim) {
        throw std::runtime_error("Dimension mismatch for vdot.");
    }
    auto bra_t = bra.dtype;
    auto ket_t = ket.dtype;
    switch (bra_t) {
        VDOT_TENSOR(TDtype::Float32)
        VDOT_TENSOR(TDtype::Float64)
        VDOT_TENSOR(TDtype::Complex64)
        VDOT_TENSOR(TDtype::Complex128)
    }
}
#undef VDOT_TENSOR

// -----------------------------------------------------------------------------

bool is_all_zero(const Tensor& t) {
    switch (t.dtype) {
        case (TDtype::Float32):
            return is_all_zero<TDtype::Float32>(t.data, t.dim);
        case (TDtype::Float64):
            return is_all_zero<TDtype::Float64>(t.data, t.dim);
        case (TDtype::Complex64):
            return is_all_zero<TDtype::Complex64>(t.data, t.dim);
        case (TDtype::Complex128):
            return is_all_zero<TDtype::Complex128>(t.data, t.dim);
    }
    return false;
}

// -----------------------------------------------------------------------------
#define IS_EQUAL_TO_TENSOR(lhs_dtype)                                                                                  \
    case (lhs_dtype): {                                                                                                \
        switch (rhs.dtype) {                                                                                           \
            case (TDtype::Float32):                                                                                    \
                return is_equal_to<lhs_dtype, TDtype::Float32>(lhs.data, lhs.dim, rhs.data, rhs.dim);                  \
            case (TDtype::Float64):                                                                                    \
                return is_equal_to<lhs_dtype, TDtype::Float64>(lhs.data, lhs.dim, rhs.data, rhs.dim);                  \
            case (TDtype::Complex64):                                                                                  \
                return is_equal_to<lhs_dtype, TDtype::Complex64>(lhs.data, lhs.dim, rhs.data, rhs.dim);                \
            case (TDtype::Complex128):                                                                                 \
                return is_equal_to<lhs_dtype, TDtype::Complex128>(lhs.data, lhs.dim, rhs.data, rhs.dim);               \
        }                                                                                                              \
        break;                                                                                                         \
    }

std::vector<bool> is_equal_to(const Tensor& lhs, const Tensor& rhs) {
    switch (lhs.dtype) {
        IS_EQUAL_TO_TENSOR(TDtype::Float32)
        IS_EQUAL_TO_TENSOR(TDtype::Float64)
        IS_EQUAL_TO_TENSOR(TDtype::Complex64)
        IS_EQUAL_TO_TENSOR(TDtype::Complex128)
    }
}
#undef IS_EQUAL_TO_TENSOR

// -----------------------------------------------------------------------------

Tensor Gather(const std::vector<Tensor>& tensors) {
    if (tensors.size() == 0) {
        return Tensor();
    }
    auto dtype = tensors[0].dtype;
    size_t tot_len = 0;
    for (auto& t : tensors) {
        if (t.dtype != dtype) {
            throw std::runtime_error("element tensor of gather should have same dtype.");
        }
        tot_len += t.dim;
    }
    auto out = init(tot_len, dtype);
    size_t idx = 0;
    for (auto& t : tensors) {
        std::memcpy(reinterpret_cast<uint8_t*>(out.data) + idx, t.data, bit_size(t.dtype) * t.dim);
        idx += bit_size(t.dtype) * t.dim;
    }
    return out;
}
}  // namespace tensor::ops::cpu
