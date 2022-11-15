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
void CPUDensityMatrixPolicyBase::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
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

auto CPUDensityMatrixPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim, std::vector<py_qs_data_t>(dim));
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

bool CPUDensityMatrixPolicyBase::IsPure(qs_data_p_t qs, index_t dim) {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs_square = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    index_t row = 0;
    index_t col = 0;
    for (index_t element = 0; element < n_elements; element++) {
        for (index_t i = 0; i < col; i++) {
            qs_square[element] += qs[IdxMap(row, i)] * std::conj(qs[IdxMap(col, i)]);
        }
        for (index_t i = col; i < row; i++) {
            qs_square[element] += qs[IdxMap(row, i)] * qs[IdxMap(i, col)];
        }
        for (index_t i = row; i < dim; i++) {
            qs_square[element] += std::conj(qs[IdxMap(i, row)]) * qs[IdxMap(i, col)];
        }
        if (abs(qs_square[element] - qs[element]) > 10e-8) {
            return false;
        }
        if (col == row) {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    return true;
}

auto CPUDensityMatrixPolicyBase::PureStateVector(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    if (!IsPure(qs, dim)) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixd density matrix to vector."));
    }
    auto qs_vector = reinterpret_cast<qs_data_p_t>(calloc(dim, sizeof(qs_data_t)));
    for (index_t i = 0; i < dim; i++) {
        qs_vector[i] = sqrt(qs[IdxMap(i, i)]);
    }
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

auto CPUDensityMatrixPolicyBase::SelfHermitanHam(const py_qs_datas_t& m, const qbits_t& objs, const qbits_t& ctrls,
                                                 index_t dim) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    for (int i = 0; i < dim; i++) {
        qs[IdxMap(i, i)] = 1;
    }
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) {
                    auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    auto q = p + mask.obj_mask;
                    qs_data_t _i_p = m[0][0] * qs[IdxMap(i, p)] + m[0][1] * qs[IdxMap(j, p)];
                    qs_data_t _j_p = m[1][0] * qs[IdxMap(i, p)] + m[1][1] * qs[IdxMap(j, p)];
                    qs[IdxMap(i, p)] = _i_p;
                    qs[IdxMap(j, p)] = _j_p;
                    qs_data_t _i_q;
                    qs_data_t _j_q;
                    if (i > q) {
                        _i_q = m[0][0] * qs[IdxMap(i, q)] + m[0][1] * qs[IdxMap(j, q)];
                        _j_q = m[1][0] * qs[IdxMap(i, q)] + m[1][1] * qs[IdxMap(j, q)];
                        qs[IdxMap(i, q)] = _i_q;
                        qs[IdxMap(j, q)] = _j_q;
                    } else {
                        _i_q = m[0][0] * std::conj(qs[IdxMap(q, i)]) + m[0][1] * qs[IdxMap(j, q)];
                        _j_q = m[1][0] * std::conj(qs[IdxMap(q, i)]) + m[1][1] * qs[IdxMap(j, q)];
                        qs[IdxMap(q, i)] = std::conj(_i_q);
                        qs[IdxMap(j, q)] = _j_q;
                    }
                }
            });
    } else {
        if (mask.ctrl_qubits.size() == 1) {
            index_t ctrl_low = 0UL;
            for (qbit_t i = 0; i < mask.ctrl_qubits[0]; i++) {
                ctrl_low = (ctrl_low << 1) + 1;
            }
            index_t first_low_mask = mask.obj_low_mask;
            index_t second_low_mask = ctrl_low;
            if (mask.obj_low_mask > ctrl_low) {
                first_low_mask = ctrl_low;
                second_low_mask = mask.obj_low_mask;
            }
            auto first_high_mask = ~first_low_mask;
            auto second_high_mask = ~second_low_mask;

            THRESHOLD_OMP_FOR(
                dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & first_high_mask) << 1) + (a & first_low_mask);
                    i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                    auto j = i + mask.obj_mask;
                    for (index_t b = 0; b <= a; b++) {
                        auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                        auto q = p + mask.obj_mask;
                        qs_data_t _i_p = m[0][0] * qs[IdxMap(i, p)] + m[0][1] * qs[IdxMap(j, p)];
                        qs_data_t _j_p = m[1][0] * qs[IdxMap(i, p)] + m[1][1] * qs[IdxMap(j, p)];
                        qs[IdxMap(i, p)] = _i_p;
                        qs[IdxMap(j, p)] = _j_p;
                        qs_data_t _i_q;
                        qs_data_t _j_q;
                        if (i > q) {
                            _i_q = m[0][0] * qs[IdxMap(i, q)] + m[0][1] * qs[IdxMap(j, q)];
                            _j_q = m[1][0] * qs[IdxMap(i, q)] + m[1][1] * qs[IdxMap(j, q)];
                            qs[IdxMap(i, q)] = _i_q;
                            qs[IdxMap(j, q)] = _j_q;
                        } else {
                            _i_q = m[0][0] * std::conj(qs[IdxMap(q, i)]) + m[0][1] * qs[IdxMap(j, q)];
                            _j_q = m[1][0] * std::conj(qs[IdxMap(q, i)]) + m[1][1] * qs[IdxMap(j, q)];

                            qs[IdxMap(q, i)] = std::conj(_i_q);
                            qs[IdxMap(j, q)] = _j_q;
                        }
                    }
                });
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        for (index_t b = 0; b <= a; b++) {
                            auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                            auto q = p + mask.obj_mask;
                            qs_data_t _i_p = m[0][0] * qs[IdxMap(i, p)] + m[0][1] * qs[IdxMap(j, p)];
                            qs_data_t _j_p = m[1][0] * qs[IdxMap(i, p)] + m[1][1] * qs[IdxMap(j, p)];
                            qs[IdxMap(i, p)] = _i_p;
                            qs[IdxMap(j, p)] = _j_p;
                            qs_data_t _i_q;
                            qs_data_t _j_q;
                            if (i > q) {
                                _i_q = m[0][0] * qs[IdxMap(i, q)] + m[0][1] * qs[IdxMap(j, q)];
                                _j_q = m[1][0] * qs[IdxMap(i, q)] + m[1][1] * qs[IdxMap(j, q)];
                                qs[IdxMap(i, q)] = _i_q;
                                qs[IdxMap(j, q)] = _j_q;
                            } else {
                                _i_q = m[0][0] * std::conj(qs[IdxMap(q, i)]) + m[0][1] * qs[IdxMap(j, q)];
                                _j_q = m[1][0] * std::conj(qs[IdxMap(q, i)]) + m[1][1] * qs[IdxMap(j, q)];
                                qs[IdxMap(q, i)] = std::conj(_i_q);
                                qs[IdxMap(j, q)] = _j_q;
                            }
                        }
                    }
                });
        }
    }
    return qs;
};

}  // namespace mindquantum::sim::densitymatrix::detail