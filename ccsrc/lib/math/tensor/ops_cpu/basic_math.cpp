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

#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
Matrix MatMul(const Matrix& m1, const Matrix& m2) {
    switch (m1.dtype) {
        case TDtype::Float32: {
            switch (m2.dtype) {
                case TDtype::Float32: {
                    return MatMul<TDtype::Float32, TDtype::Float32>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                    m2.n_col);
                }
                case TDtype::Float64: {
                    return MatMul<TDtype::Float32, TDtype::Float64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                    m2.n_col);
                }
                case TDtype::Complex64: {
                    return MatMul<TDtype::Float32, TDtype::Complex64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                      m2.n_col);
                }
                case TDtype::Complex128: {
                    return MatMul<TDtype::Float32, TDtype::Complex128>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                       m2.n_col);
                }
            }
        } break;
        case TDtype::Float64: {
            switch (m2.dtype) {
                case TDtype::Float32: {
                    return MatMul<TDtype::Float64, TDtype::Float32>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                    m2.n_col);
                }
                case TDtype::Float64: {
                    return MatMul<TDtype::Float64, TDtype::Float64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                    m2.n_col);
                }
                case TDtype::Complex64: {
                    return MatMul<TDtype::Float64, TDtype::Complex64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                      m2.n_col);
                }
                case TDtype::Complex128: {
                    return MatMul<TDtype::Float64, TDtype::Complex128>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                       m2.n_col);
                }
            }
        } break;
        case TDtype::Complex64: {
            switch (m2.dtype) {
                case TDtype::Float32: {
                    return MatMul<TDtype::Complex64, TDtype::Float32>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                      m2.n_col);
                }
                case TDtype::Float64: {
                    return MatMul<TDtype::Complex64, TDtype::Float64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                      m2.n_col);
                }
                case TDtype::Complex64: {
                    return MatMul<TDtype::Complex64, TDtype::Complex64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                        m2.n_col);
                }
                case TDtype::Complex128: {
                    return MatMul<TDtype::Complex64, TDtype::Complex128>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                         m2.n_col);
                }
            }
        } break;
        case TDtype::Complex128: {
            switch (m2.dtype) {
                case TDtype::Float32: {
                    return MatMul<TDtype::Complex128, TDtype::Float32>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                       m2.n_col);
                }
                case TDtype::Float64: {
                    return MatMul<TDtype::Complex128, TDtype::Float64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                       m2.n_col);
                }
                case TDtype::Complex64: {
                    return MatMul<TDtype::Complex128, TDtype::Complex64>(m1.data, m1.n_row, m1.n_col, m2.data, m2.n_row,
                                                                         m2.n_col);
                }
                case TDtype::Complex128: {
                    return MatMul<TDtype::Complex128, TDtype::Complex128>(m1.data, m1.n_row, m1.n_col, m2.data,
                                                                          m2.n_row, m2.n_col);
                }
            }
        } break;
    }
}
}  // namespace tensor::ops::cpu
