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
#include <iostream>
#include <string>

#include "core/utils.h"
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
}  // namespace mindquantum
#endif  // MINDQUANTUM_MATRIX_TWO_DIM_MATRIX_H_
