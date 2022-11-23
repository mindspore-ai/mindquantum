//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {

// Warning: only correct when x >= y
index_t CPUDensityMatrixPolicyBase::IdxMap(index_t x, index_t y) {
    return (x * x + x) / 2 + y;
}

auto CPUDensityMatrixPolicyBase::GetValue(qs_data_p_t qs, index_t x, index_t y) -> qs_data_t {
    if (x >= y) {
        return qs[IdxMap(x, y)];
    } else {
        return std::conj(qs[IdxMap(y, x)]);
    }
}

auto CPUDensityMatrixPolicyBase::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

void CPUDensityMatrixPolicyBase::Reset(qs_data_p_t qs, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = 0; })
    qs[0] = 1;
}

void CPUDensityMatrixPolicyBase::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

// need to fix
void CPUDensityMatrixPolicyBase::Display(py_qs_datas_t& qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
    }
}

void CPUDensityMatrixPolicyBase::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                for (index_t j = 0; j < (dim / 2); j++) {
                    if ((j & ctrl_mask) != ctrl_mask) {
                        qs[IdxMap(i, j)] = 0;
                    }
                }
            }
        })
}

auto CPUDensityMatrixPolicyBase::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = CPUDensityMatrixPolicyBase::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { out[i] = qs[i]; })
    return out;
}

auto CPUDensityMatrixPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> matrix_t {
    matrix_t out(dim, py_qs_datas_t(dim));
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j < i; j++) {
                out[i][j] = qs[IdxMap(i, j)];
            }
            for (index_t j = i; j < dim; j++) {
                out[i][j] = std::conj(qs[IdxMap(j, i)]);
            }
        })
    return out;
}

void CPUDensityMatrixPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i]; })
}

void CPUDensityMatrixPolicyBase::SetQS(qs_data_p_t qs, const qs_data_p_t qs_out, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = qs_out[i]; })
}

auto CPUDensityMatrixPolicyBase::Purity(qs_data_p_t qs, index_t dim) -> calc_type {
    calc_type p = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { p += 2 * std::norm(qs[i]); })
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < dim; i++) { p += -std::norm(qs[IdxMap(i, i)]); })
    return p;
}

bool CPUDensityMatrixPolicyBase::IsPure(qs_data_p_t qs, index_t dim) {
    auto p = Purity(qs, dim);
    if (std::abs(p - 1) < 1e-8) {
        return true;
    } else {
        return false;
    }
}

auto CPUDensityMatrixPolicyBase::PureStateVector(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    if (!IsPure(qs, dim)) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixd density matrix to vector."));
    }
    py_qs_datas_t qs_vector(dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs_vector[i] = std::sqrt(qs[IdxMap(i, i)]); })
    return qs_vector;
}
// auto CPUDensityMatrixPolicyBase::MatrixMul(qs_data_p_t qs, )

void CPUDensityMatrixPolicyBase::DisplayQS(qs_data_p_t qs, qbit_t n_qubits, index_t dim) {
    auto out = CPUDensityMatrixPolicyBase::GetQS(qs, dim);
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < dim; i++) {
        for (index_t j = 0; j < dim; j++) {
            std::cout << "(" << out[i][j].real() << ", " << out[i][j].imag() << ")"
                      << ",";
        }
        std::cout << std::endl;
    }
}

// // need to fix
// void CPUDensityMatrixPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
//     if (qs_out.size() != dim) {
//         throw std::invalid_argument("state size not match");
//     }
//     THRESHOLD_OMP_FOR(
//         dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i][i]; })
// }

auto CPUDensityMatrixPolicyBase::DiagonalConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs,
                                                            index_t dim) -> calc_type {
    // collect diagonal amplitude with index mask satisfied condition.
    calc_type res_real = 0;
    if (abs) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 auto _ii = IdxMap(i, i);
                                 res_real += qs[_ii].real();
                             }
                         });
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[IdxMap(i, i)].real();
                             }
                         });
    }
    return res_real;
}

template <class binary_op>
void CPUDensityMatrixPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                   qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                                   const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
            auto _i_0 = IdxMap(i, 0);
            if ((i & mask) == condi) {
                for (index_t j = 0; j <= i; j++) {
                    if ((j & mask) == condi) {
                        des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                    } else {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            } else {
                for (index_t j = 0; j <= i; j++) {
                    des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                }
            }
        })
}

template <index_t mask, index_t condi, class binary_op>
void CPUDensityMatrixPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                                   qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
            auto _i_0 = IdxMap(i, 0);
            if ((i & mask) == condi) {
                for (index_t j = 0; j <= i; j++) {
                    if ((j & mask) == condi) {
                        des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                    } else {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            } else {
                for (index_t j = 0; j <= i; j++) {
                    des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                }
            }
        })
}

void CPUDensityMatrixPolicyBase::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, std::multiplies<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                  qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::divides<qs_data_t>());
}

void CPUDensityMatrixPolicyBase::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val) * IMAGE_MI;
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val) * IMAGE_MI;
    }
    VT<VT<qs_data_t>> m{{c, 0, 0, s}, {0, c, s, 0}, {0, s, c, 0}, {s, 0, 0, c}};
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
                VT<index_t> row(4);  // row index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                              row[0]);
                row[3] = row[0] + mask.obj_mask;
                row[1] = row[0] + mask.obj_min_mask;
                row[2] = row[0] + mask.obj_max_mask;
                for (index_t b = 0; b <= a; b++) {
                    VT<index_t> col(4);  // column index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    col[3] = col[0] + mask.obj_mask;
                    col[1] = col[0] + mask.obj_min_mask;
                    col[2] = col[0] + mask.obj_max_mask;

                    VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = c * GetValue(qs, row[i], col[j]) + s * GetValue(qs, row[3 - i], col[j]);
                        }
                    }
                    auto conj_s = std::conj(s);
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            if (row[i] >= col[j]) {
                                qs[IdxMap(row[i], col[j])] = c * tmp_mat[i][j] + conj_s * tmp_mat[i][3 - j];
                            } else {
                                qs[IdxMap(row[j], col[i])] = std::conj(c * tmp_mat[i][j] + conj_s * tmp_mat[i][3 - j]);
                            }
                        }
                    }
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
                VT<index_t> row(4);  // row index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                              row[0]);
                row[3] = row[0] + mask.obj_mask;
                row[1] = row[0] + mask.obj_min_mask;
                row[2] = row[0] + mask.obj_max_mask;
                for (index_t b = 0; b <= a; b++) {
                    VT<index_t> col(4);  // column index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, col[0]);
                    if ((row[0] & mask.ctrl_mask) != mask.ctrl_mask) {
                        if ((col[0] & mask.ctrl_mask) != mask.ctrl_mask) {  // both not in control
                            continue;
                        }
                        col[3] = col[0] + mask.obj_mask;
                        col[1] = col[0] + mask.obj_min_mask;
                        col[2] = col[0] + mask.obj_max_mask;
                        VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                        if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                            for (int i = 0; i < 4; i++) {
                                for (int j = 0; j < 4; j++) {
                                    tmp_mat[i][j] = c * GetValue(qs, row[i], col[j])
                                                    + s * GetValue(qs, row[3 - i], col[j]);
                                }
                            }
                        } else {  // row not in control
                            for (int i = 0; i < 4; i++) {
                                for (int j = 0; j < 4; j++) {
                                    tmp_mat[i][j] = GetValue(qs, row[i], col[j]);
                                }
                            }
                        }
                        auto conj_s = std::conj(s);
                        if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                            for (int i = 0; i < 4; i++) {
                                for (int j = 0; j < 4; j++) {
                                    if (row[i] >= col[j]) {
                                        qs[IdxMap(row[i], col[j])] = c * tmp_mat[i][j] + conj_s * tmp_mat[i][3 - j];
                                    } else {
                                        qs[IdxMap(row[j], col[i])] = std::conj(c * tmp_mat[i][j]
                                                                               + conj_s * tmp_mat[i][3 - j]);
                                    }
                                }
                            }
                        } else {  // column not in control
                            for (int i = 0; i < 4; i++) {
                                for (int j = 0; j < 4; j++) {
                                    if (row[i] >= col[j]) {
                                        qs[IdxMap(row[i], col[j])] = tmp_mat[i][j];
                                    } else {
                                        qs[IdxMap(row[j], col[i])] = std::conj(tmp_mat[i][j]);
                                    }
                                }
                            }
                        }
                    }
                }
            })
        if (diff) {
            CPUDensityMatrixPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

}  // namespace mindquantum::sim::densitymatrix::detail