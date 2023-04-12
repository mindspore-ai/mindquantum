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

#include "math/tensor/csr_matrix.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
#define MM_MatMul(m1_dtype)                                                                                            \
    case m1_dtype: {                                                                                                   \
        switch (m2.dtype) {                                                                                            \
            case TDtype::Float32: {                                                                                    \
                return MatMul<m1_dtype, TDtype::Float32>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row, m2.n_col);    \
            }                                                                                                          \
            case TDtype::Float64: {                                                                                    \
                return MatMul<m1_dtype, TDtype::Float64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row, m2.n_col);    \
            }                                                                                                          \
            case TDtype::Complex64: {                                                                                  \
                return MatMul<m1_dtype, TDtype::Complex64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row, m2.n_col);  \
            }                                                                                                          \
            case TDtype::Complex128: {                                                                                 \
                return MatMul<m1_dtype, TDtype::Complex128>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row, m2.n_col); \
            }                                                                                                          \
        }                                                                                                              \
        break;                                                                                                         \
    }
Matrix MatMul(const Matrix& m1, const Matrix& m2) {
    switch (m1.dtype) {
        MM_MatMul(TDtype::Float32);
        MM_MatMul(TDtype::Float64);
        MM_MatMul(TDtype::Complex64);
        MM_MatMul(TDtype::Complex128);
    }
}
#undef MM_MatMul

// -----------------------------------------------------------------------------
#define CSR_MatMul(m1_dtype)                                                                                           \
    case m1_dtype: {                                                                                                   \
        switch (m2.dtype) {                                                                                            \
            case TDtype::Float32: {                                                                                    \
                return MatMul<m1_dtype, TDtype::Float32>(m1.data_.data, m1.indptr_, m1.indices_, m1.n_row, m1.n_col,   \
                                                         m1.nnz, m2.data, m2.dim);                                     \
            }                                                                                                          \
            case TDtype::Float64: {                                                                                    \
                return MatMul<m1_dtype, TDtype::Float64>(m1.data_.data, m1.indptr_, m1.indices_, m1.n_row, m1.n_col,   \
                                                         m1.nnz, m2.data, m2.dim);                                     \
            }                                                                                                          \
            case TDtype::Complex64: {                                                                                  \
                return MatMul<m1_dtype, TDtype::Complex64>(m1.data_.data, m1.indptr_, m1.indices_, m1.n_row, m1.n_col, \
                                                           m1.nnz, m2.data, m2.dim);                                   \
            }                                                                                                          \
            case TDtype::Complex128: {                                                                                 \
                return MatMul<m1_dtype, TDtype::Complex128>(m1.data_.data, m1.indptr_, m1.indices_, m1.n_row,          \
                                                            m1.n_col, m1.nnz, m2.data, m2.dim);                        \
            }                                                                                                          \
        }                                                                                                              \
        break;                                                                                                         \
    }

Tensor MatMul(const CsrMatrix& m1, const Tensor& m2) {
    switch (m1.GetDtype()) {
        CSR_MatMul(TDtype::Float32);
        CSR_MatMul(TDtype::Float64);
        CSR_MatMul(TDtype::Complex64);
        CSR_MatMul(TDtype::Complex128);
    }
}
#undef CSR_MatMul
}  // namespace tensor::ops::cpu
