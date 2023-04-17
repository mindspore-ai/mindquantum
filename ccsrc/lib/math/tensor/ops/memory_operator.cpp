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

#include "math/tensor/ops/memory_operator.hpp"

#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops {
Tensor init(size_t len, TDtype dtype, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init(len, dtype);
    } else {
    }
}

Tensor cast_to(const Tensor& t, TDtype target_dtype) {
    if (t.device == TDevice::CPU) {
        return cpu::cast_to(t, target_dtype);
    }
}

std::string to_string(const Tensor& t, bool simplify) {
    if (t.device == TDevice::CPU) {
        return cpu::to_string(t, simplify);
    }
}

void destroy(Tensor* t) {
    if (t->device == TDevice::CPU) {
        cpu::destroy(t);
    } else {
    }
}

// -----------------------------------------------------------------------------

Tensor init_with_value(float a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_value(a);
    } else {
    }
}

Tensor init_with_value(double a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_value(a);
    } else {
    }
}

Tensor init_with_value(const std::complex<float>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_value(a);
    } else {
    }
}

Tensor init_with_value(const std::complex<double>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_value(a);
    } else {
    }
}

Tensor init_with_vector(const std::vector<float>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_vector(a);
    } else {
    }
}
Tensor init_with_vector(const std::vector<double>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_vector(a);
    } else {
    }
}
Tensor init_with_vector(const std::vector<std::complex<float>>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_vector(a);
    } else {
    }
}
Tensor init_with_vector(const std::vector<std::complex<double>>& a, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::init_with_vector(a);
    } else {
    }
}

// -----------------------------------------------------------------------------

Tensor copy(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        return cpu::copy(t);
    } else {
    }
}

// -----------------------------------------------------------------------------

void set(Tensor* t, float a, size_t idx) {
    if (t->device == TDevice::CPU) {
        cpu::set(t->data, t->dtype, a, t->dim, idx);
    } else {
    }
}
void set(Tensor* t, double a, size_t idx) {
    if (t->device == TDevice::CPU) {
        cpu::set(t->data, t->dtype, a, t->dim, idx);
    } else {
    }
}
void set(Tensor* t, const std::complex<float>& a, size_t idx) {
    if (t->device == TDevice::CPU) {
        cpu::set(t->data, t->dtype, a, t->dim, idx);
    } else {
    }
}
void set(Tensor* t, const std::complex<double>& a, size_t idx) {
    if (t->device == TDevice::CPU) {
        cpu::set(t->data, t->dtype, a, t->dim, idx);
    } else {
    }
}

void set(Tensor* t, const Tensor& source, size_t idx) {
    if (t->device == TDevice::CPU) {
        cpu::set(t, source, idx);
    }
}

// -----------------------------------------------------------------------------

Tensor get(const Tensor& t, size_t idx) {
    if (t.device == TDevice::GPU) {
        return cpu::get(t, idx);
    } else {
    }
}

}  // namespace tensor::ops

namespace tensor {
Tensor::~Tensor() {
    ops::destroy(this);
}
Tensor::Tensor(float a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_value(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(double a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_value(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::complex<float>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_value(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::complex<double>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_value(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::vector<float>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_vector(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::vector<double>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_vector(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::vector<std::complex<float>>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_vector(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(const std::vector<std::complex<double>>& a, TDtype dtype) {
    auto tmp = tensor::ops::init_with_vector(a).astype(dtype);
    this->data = tmp.data;
    this->dtype = tmp.dtype;
    this->device = tmp.device;
    this->dim = tmp.dim;
    tmp.data = nullptr;
}
Tensor::Tensor(TDtype dtype, TDevice device, void* data, size_t dim)
    : dtype(dtype), device(device), data(data), dim(dim) {
}

Tensor::Tensor(Tensor&& t) {
    this->data = t.data;
    t.data = nullptr;
    this->dim = t.dim;
    this->device = t.device;
    this->dtype = t.dtype;
}
Tensor& Tensor::operator=(Tensor&& t) {
    ops::destroy(this);
    this->data = t.data;
    t.data = nullptr;
    this->dim = t.dim;
    this->device = t.device;
    this->dtype = t.dtype;
    return *this;
}
Tensor& Tensor::operator=(const Tensor& t) {
    ops::destroy(this);
    if (t.device == TDevice::CPU) {
        this->data = ops::cpu::copy_mem(t.data, t.dtype, t.dim);
    } else {
    }
    this->device = t.device;
    this->dtype = t.dtype;
    this->dim = t.dim;
    return *this;
}

Tensor::Tensor(const Tensor& t) {
    if (t.device == TDevice::CPU) {
        this->data = ops::cpu::copy_mem(t.data, t.dtype, t.dim);
    } else {
    }
    this->device = t.device;
    this->dtype = t.dtype;
    this->dim = t.dim;
}
}  // namespace tensor

std::ostream& operator<<(std::ostream& os, const tensor::Tensor& t) {
    os << tensor::ops::to_string(t);
    return os;
}
