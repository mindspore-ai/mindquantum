/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "math/tensor/ops/advance_math.h"

#include <cmath>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "math/tensor/ops_cpu/advance_math.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops {
Tensor sin(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::ElementFunc(t, t.dtype, [](auto i) { return std::sin(i); });
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor cos(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::ElementFunc(t, t.dtype, [](auto i) { return std::cos(i); });
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor exp(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::ElementFunc(t, t.dtype, [](auto i) { return std::exp(i); });
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor sqrt(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::ElementFunc(t, t.dtype, [](auto i) { return std::sqrt(i); });
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor gather(const std::vector<Tensor>& tensors) {
    if (tensors.size() == 0) {
        return Tensor();
    }
    auto device = tensors[0].device;
    if (std::any_of(tensors.begin(), tensors.end(), [&](const auto& t) { return t.device != device; })) {
        throw std::runtime_error("Gather only work for tensor in save device.");
    }
    if (device == TDevice::CPU) {
        return ops::cpu::Gather(tensors);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor real(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::real(t);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}

Tensor imag(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::imag(t);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}
Tensor conj(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::conj(t);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}
Tensor vdot(const Tensor& bra, const Tensor& ket) {
    if (bra.device != ket.device) {
        throw std::runtime_error("Cannot vdot between to difference kind of device.");
    }
    if (bra.device == TDevice::CPU) {
        return ops::cpu::vdot(bra, ket);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return Tensor();
}
bool is_all_zero(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::is_all_zero(t);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return false;
}
std::vector<bool> is_equal_to(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.device != rhs.device) {
        throw std::runtime_error("Cannot compare tow tensor in different device.");
    }
    if (lhs.device == TDevice::CPU) {
        return ops::cpu::is_equal_to(lhs, rhs);
    } else {
        throw std::runtime_error("No support GPU now.");
    }
    return std::vector<bool>(lhs.dim, false);
}
}  // namespace tensor::ops
