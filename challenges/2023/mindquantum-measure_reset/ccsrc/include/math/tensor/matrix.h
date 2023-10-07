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
#ifndef MATH_TENSOR_MATRIX
#define MATH_TENSOR_MATRIX

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor {
struct Matrix : public Tensor {
    size_t n_row = 0;  // hang
    size_t n_col = 0;  // lie

    // -----------------------------------------------------------------------------

    template <typename T>
    explicit Matrix(const std::vector<std::vector<T>>& m, TDevice device = TDevice::CPU) {
        n_row = m.size();
        if (n_row != 0) {
            n_col = m[0].size();
            for (auto& row : m) {
                if (row.size() != n_col) {
                    throw std::runtime_error("Given data is not a matrix.");
                }
            }
        }
        std::vector<T> tmp;
        for (auto& row : m) {
            std::copy(row.begin(), row.end(), std::back_inserter(tmp));
        }
        auto t = Tensor(tmp);
        this->dtype = t.dtype;
        this->device = device;
        this->data = t.data;
        this->dim = t.dim;
        t.data = nullptr;
    }
    Matrix() = default;
    Matrix(TDtype dtype, TDevice device, void* data, size_t n_row, size_t n_col)
        : Tensor(dtype, device, data, n_col * n_row), n_row(n_row), n_col(n_col) {
    }
    Matrix(Tensor&& other, size_t n_row, size_t n_col) : n_row(n_row), n_col(n_col) {
        if (n_col * n_row != other.dim) {
            throw std::runtime_error("Tensor cannot reshape to Matrix with given n_col and n_row.");
        }
        this->dtype = other.dtype;
        this->device = other.device;
        this->data = other.data;
        this->dim = other.dim;
        other.data = nullptr;
    }
};
}  // namespace tensor
#endif /* MATH_TENSOR_MATRIX */
