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

#include <stdexcept>

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

}  // namespace tensor::ops::cpu
