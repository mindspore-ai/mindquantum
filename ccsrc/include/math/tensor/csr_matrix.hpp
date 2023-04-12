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

#ifndef MATH_TENSOR_CSR_MATRIX_HPP_
#define MATH_TENSOR_CSR_MATRIX_HPP_
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"
namespace tensor {
struct CsrMatrix {
    size_t n_row = 0;
    size_t n_col = 0;
    size_t nnz = 0;
    size_t* indptr_ = nullptr;
    size_t* indices_ = nullptr;
    Tensor data_{};

    // -----------------------------------------------------------------------------
    CsrMatrix() = default;
    CsrMatrix(CsrMatrix&& t);
    CsrMatrix& operator=(CsrMatrix&& t);
    CsrMatrix(const CsrMatrix& t);
    CsrMatrix& operator=(const CsrMatrix& t);
    CsrMatrix(size_t n_row, size_t n_col, size_t nnz, size_t* indptr, size_t* indices, const Tensor& data)
        : n_row(n_row), n_col(n_col), nnz(nnz), indptr_(indptr), indices_(indices), data_(data) {
    }
    CsrMatrix(size_t n_row, size_t n_col, size_t nnz, TDtype dtype);
    ~CsrMatrix() {
        if (indices_ != nullptr) {
            free(indices_);
            indices_ = nullptr;
        }
        if (indptr_ != nullptr) {
            free(indptr_);
            indptr_ = nullptr;
        }
    }

    // -----------------------------------------------------------------------------

    TDtype GetDtype() const;
    void CastTo(TDtype dtype);
    CsrMatrix hermitian_conjugated();
    
};
}  // namespace tensor
#endif /* MATH_TENSOR_CSR_MATRIX_HPP_ */
