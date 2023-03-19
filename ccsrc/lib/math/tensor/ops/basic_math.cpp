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

#include "math/tensor/ops_cpu/basic_math.hpp"

#include <functional>

#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/traits.hpp"
namespace tensor::ops {
void inplace_add(Tensor* t, float a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<float, std::plus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}
void inplace_add(Tensor* t, double a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<double, std::plus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}
void inplace_add(Tensor* t, const std::complex<float>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<float>, std::plus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}
void inplace_add(Tensor* t, const std::complex<double>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<double>, std::plus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

Tensor add(const Tensor& t, float a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<float, std::plus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor add(const Tensor& t, double a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<double, std::plus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor add(const Tensor& t, const std::complex<float>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<float>, std::plus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor add(const Tensor& t, const std::complex<double>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<double>, std::plus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}

// -----------------------------------------------------------------------------

void inplace_sub(Tensor* t, float a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<float, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(Tensor* t, double a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<double, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(Tensor* t, const std::complex<float>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<float>, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(Tensor* t, const std::complex<double>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<double>, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(float a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<float, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(double a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<double, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(const std::complex<float>& a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<std::complex<float>, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_sub(const std::complex<double>& a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<std::complex<double>, std::minus>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

Tensor sub(const Tensor& t, float a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<float, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(const Tensor& t, double a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<double, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(const Tensor& t, const std::complex<float>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<float>, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(const Tensor& t, const std::complex<double>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<double>, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}

Tensor sub(float a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<float, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(double a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<double, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(const std::complex<float>& a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<std::complex<float>, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor sub(const std::complex<double>& a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<std::complex<double>, std::minus>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
// -----------------------------------------------------------------------------

void inplace_mul(Tensor* t, float a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<float, std::multiplies>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_mul(Tensor* t, double a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<double, std::multiplies>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_mul(Tensor* t, const std::complex<float>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<float>, std::multiplies>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_mul(Tensor* t, const std::complex<double>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<double>, std::multiplies>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

Tensor mul(const Tensor& t, float a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<float, std::multiplies>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor mul(const Tensor& t, double a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<double, std::multiplies>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor mul(const Tensor& t, const std::complex<float>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<float>, std::multiplies>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor mul(const Tensor& t, const std::complex<double>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<double>, std::multiplies>(t.data, t.dtype, t.dim, a);
    } else {
    }
}

// -----------------------------------------------------------------------------

void inplace_div(Tensor* t, float a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<float, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(Tensor* t, double a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<double, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(Tensor* t, const std::complex<float>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<float>, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(Tensor* t, const std::complex<double>& a) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary<std::complex<double>, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}
void inplace_div(float a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<float, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(double a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<double, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(const std::complex<float>& a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<std::complex<float>, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

void inplace_div(const std::complex<double>& a, Tensor* t) {
    if (t->device == TDevice::CPU) {
        ops::cpu::inplace_binary_rev<std::complex<double>, std::divides>(t->data, t->dtype, t->dim, a);
    } else {
    }
}

Tensor div(const Tensor& t, float a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<float, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(const Tensor& t, double a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<double, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(const Tensor& t, const std::complex<float>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<float>, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(const Tensor& t, const std::complex<double>& a) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary<std::complex<double>, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}

Tensor div(float a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<float, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(double a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<double, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(const std::complex<float>& a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<std::complex<float>, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
Tensor div(const std::complex<double>& a, const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return ops::cpu::tensor_binary_rev<std::complex<double>, std::divides>(t.data, t.dtype, t.dim, a);
    } else {
    }
}
}  // namespace tensor::ops
