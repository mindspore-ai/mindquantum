/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDQUANTUM_MATRIX_TWO_DIM_MATRIX_H_
#define MINDQUANTUM_MATRIX_TWO_DIM_MATRIX_H_
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

#include "core/utils.hpp"
namespace mindquantum {
template <typename T>
struct Dim2Matrix {
    VVT<CT<T>> matrix_;
    Dim2Matrix() {
    }
    explicit Dim2Matrix(const VVT<CT<T>> &m) : matrix_(m) {
    }
    void PrintInfo() {
        if (matrix_.size() > 0) {
            if (matrix_[0].size() > 0) {
                std::cout << "<--Matrix of " << matrix_.size() << " X " << matrix_[0].size() << std::endl;
                for (auto &col : matrix_) {
                    for (Index i = 0; i < static_cast<Index>(col.size()); i++) {
                        std::cout << col[i];
                        if (i != static_cast<Index>(col.size() - 1)) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << std::endl;
                }
                std::cout << "-->" << std::endl;
            }
        }
    }
};

template <typename T, class binary_op>
void Dim2MatrixBinary(Dim2Matrix<T> *m, CT<T> val, const binary_op &op) {
    for (auto &col : m->matrix_) {
        for (auto &i : col) {
            i = op(i, val);
        }
    }
}

template <typename T>
Dim2Matrix<T> Dim2MatrixFromRI(const VT<VS> &real, const VT<VS> &imag) {
    Dim2Matrix<T> out;
    for (Index i = 0; i < static_cast<Index>(real.size()); i++) {
        out.matrix_.push_back({});
        for (Index j = 0; j < static_cast<Index>(real[i].size()); j++) {
            out.matrix_[i].push_back(CT<T>(std::stod(real[i][j]), std::stod(imag[i][j])));
        }
    }
    return out;
}

// TODO(xuxs): In the next version, we will use eigen for all matrix element. But
// currently, we just custom some mathmatic operator for Dim2Matrix.

template <typename T, class binary_ops>
Dim2Matrix<T> Dim2MatrixBinary(const Dim2Matrix<T> &m1, const Dim2Matrix<T> &m2, const binary_ops &ops) {
    assert(m1.matrix_.size() == m2.matrix_.size());
    VVT<CT<T>> m(m1.matrix_.size(), {});
    for (size_t i = 0; i < m1.matrix_.size(); i++) {
        assert(m1.matrix_[i].size() == m2.matrix_.size());
        for (size_t j = 0; j < m1.matrix_[i].size(); j++) {
            m[i].push_back(ops(m1.matrix_[i][j], m2.matrix_[i][j]));
        }
    }
    return Dim2Matrix<T>(m);
}
template <typename T>
Dim2Matrix<T> Dim2MatrixMatMul(const Dim2Matrix<T> &m1, const Dim2Matrix<T> &m2) {
    // Be carefule! We will check dimension.
    VVT<CT<T>> m(m1.matrix_.size(), VT<CT<T>>(m2.matrix_[0].size(), CT<T>(0.0, 0.0)));
    for (size_t i = 0; i < m1.matrix_.size(); i++) {
        for (size_t k = 0; k < m2.matrix_[0].size(); k++) {
            for (size_t j = 0; j < m1.matrix_[0].size(); j++) {
                m[i][k] += m1.matrix_[i][j] * m2.matrix_[j][k];
            }
        }
    }
    return Dim2Matrix<T>(m);
}
}  // namespace mindquantum
#endif  // MINDQUANTUM_MATRIX_TWO_DIM_MATRIX_H_
