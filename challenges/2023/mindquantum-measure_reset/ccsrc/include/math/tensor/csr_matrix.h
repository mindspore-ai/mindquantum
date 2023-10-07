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

#ifndef MATH_TENSOR_CSR_MATRIX_HPP_
#define MATH_TENSOR_CSR_MATRIX_HPP_
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
namespace tensor {
struct CsrMatrix {
    size_t n_row = 0;
    size_t n_col = 0;
    size_t nnz = 0;
    size_t* indptr_ = nullptr;   // size is n_row + 1
    size_t* indices_ = nullptr;  // size is nnz
    Tensor data_{};

    // -----------------------------------------------------------------------------
    CsrMatrix() = default;
    CsrMatrix(CsrMatrix&& t);
    CsrMatrix& operator=(CsrMatrix&& t);
    CsrMatrix(const CsrMatrix& t);
    CsrMatrix& operator=(const CsrMatrix& t);
    CsrMatrix(size_t n_row, size_t n_col, size_t nnz, size_t* indptr, size_t* indices, Tensor&& data);
    ~CsrMatrix();

    // -----------------------------------------------------------------------------

    TDtype GetDtype() const;
    TDevice GetDevice() const;
    void CastTo(TDtype dtype);
};
}  // namespace tensor
#endif /* MATH_TENSOR_CSR_MATRIX_HPP_ */
