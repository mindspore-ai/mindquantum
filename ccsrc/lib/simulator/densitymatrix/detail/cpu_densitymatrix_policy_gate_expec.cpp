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
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {

auto CPUDensityMatrixPolicyBase::HamiltonianMatrix(const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
    qs_data_p_t out = InitState(dim, false);
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                    auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                    auto c = POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)];
                    out[IdxMap(j, i)] = coeff * c;
                }
            })
    }
    return out;
}

auto CPUDensityMatrixPolicyBase::GetExpectation(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim) -> qs_data_t {
    qs_data_t expectation_value{0};
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                    auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                    auto c = POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)];
                    expectation_value += std::conj(qs[IdxMap(j, i)]) * coeff * c;
                    if (i != j) {
                        expectation_value += qs[IdxMap(j, i)] * coeff / c;
                    }
                }
            })
    }
    return expectation_value;
}

auto CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                             const qbits_t& objs, const qbits_t& ctrls,
                                                             const matrix_t& m, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t res;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {
                    auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs_data_t _i_col;
                    qs_data_t _j_col;
                    for (index_t col = 0; col < dim; col++) {
                        if (i >= col) {
                            _i_col = m[0][0] * qs[IdxMap(i, col)] + m[0][1] * qs[IdxMap(j, col)];
                            _j_col = m[1][0] * qs[IdxMap(i, col)] + m[1][1] * qs[IdxMap(j, col)];
                            res += _i_col * std::conj(ham_matrix[IdxMap(i, col)])
                                   + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
                        } else if (j < col) {
                            _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)]) + m[0][1] * std::conj(qs[IdxMap(col, j)]);
                            _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)]) + m[1][1] * std::conj(qs[IdxMap(col, j)]);
                            res += _i_col * ham_matrix[IdxMap(col, i)] + _j_col * ham_matrix[IdxMap(col, j)];
                        } else {
                            _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)]) + m[0][1] * qs[IdxMap(j, col)];
                            _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)]) + m[1][1] * qs[IdxMap(j, col)];
                            res += _i_col * ham_matrix[IdxMap(col, i)] + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
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

            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                        auto i = ((a & first_high_mask) << 1) + (a & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        qs_data_t _i_col;
                        qs_data_t _j_col;
                        for (index_t col = 0; col < dim; col++) {
                            if (i >= col) {
                                _i_col = m[0][0] * qs[IdxMap(i, col)] + m[0][1] * qs[IdxMap(j, col)];
                                _j_col = m[1][0] * qs[IdxMap(i, col)] + m[1][1] * qs[IdxMap(j, col)];
                                res += _i_col * std::conj(ham_matrix[IdxMap(i, col)])
                                       + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
                            } else if (j < col) {
                                _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)])
                                         + m[0][1] * std::conj(qs[IdxMap(col, j)]);
                                _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)])
                                         + m[1][1] * std::conj(qs[IdxMap(col, j)]);
                                res += _i_col * ham_matrix[IdxMap(col, i)] + _j_col * ham_matrix[IdxMap(col, j)];
                            } else {
                                _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)]) + m[0][1] * qs[IdxMap(j, col)];
                                _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)]) + m[1][1] * qs[IdxMap(j, col)];
                                res += _i_col * ham_matrix[IdxMap(col, i)]
                                       + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
                            }
                        }
                    });
        } else {
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res) schedule(static)), dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                        auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            qs_data_t _i_col;
                            qs_data_t _j_col;
                            for (index_t col = 0; col < dim; col++) {
                                if (i >= col) {
                                    _i_col = m[0][0] * qs[IdxMap(i, col)] + m[0][1] * qs[IdxMap(j, col)];
                                    _j_col = m[1][0] * qs[IdxMap(i, col)] + m[1][1] * qs[IdxMap(j, col)];
                                    res += _i_col * std::conj(ham_matrix[IdxMap(i, col)])
                                           + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
                                } else if (j < col) {
                                    _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)])
                                             + m[0][1] * std::conj(qs[IdxMap(col, j)]);
                                    _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)])
                                             + m[1][1] * std::conj(qs[IdxMap(col, j)]);
                                    res += _i_col * ham_matrix[IdxMap(col, i)] + _j_col * ham_matrix[IdxMap(col, j)];
                                } else {
                                    _i_col = m[0][0] * std::conj(qs[IdxMap(col, i)]) + m[0][1] * qs[IdxMap(j, col)];
                                    _j_col = m[1][0] * std::conj(qs[IdxMap(col, i)]) + m[1][1] * qs[IdxMap(j, col)];
                                    res += _i_col * ham_matrix[IdxMap(col, i)]
                                           + _j_col * std::conj(ham_matrix[IdxMap(j, col)]);
                                }
                            }
                        }
                    });
        }
    }
    return res;
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    matrix_t gate = {{0, 0.5 * IMAGE_MI}, {0.5 * IMAGE_MI, 0}};
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    matrix_t gate = {{0, -0.5}, {0.5, 0}};
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

auto CPUDensityMatrixPolicyBase::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                              const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t s = std::sin(val);
    qs_data_t c = std::cos(val);
    qs_data_t e0 = 0.5 * IMAGE_MI * c - 0.5 * s;
    qs_data_t e1 = 0.5 * IMAGE_I * c - 0.5 * s;
    matrix_t gate = {{e0, 0}, {0, e1}};
    return CPUDensityMatrixPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

}  // namespace mindquantum::sim::densitymatrix::detail
