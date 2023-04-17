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
#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

namespace mindquantum::sim::densitymatrix::detail {
// Single qubit operator
// ========================================================================================================

// method is based on 'mq_vector' simulator, extended to densitymatrix
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                              qbit_t obj_qubit, const qbits_t& ctrls,
                                                                              const matrix_t& m, index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {  // loop on the row
                auto r0 = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto r1 = r0 + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) {  // loop on the column
                    auto c0 = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    auto c1 = c0 + mask.obj_mask;
                    qs_data_t src_00 = src[IdxMap(r0, c0)];
                    qs_data_t src_11 = src[IdxMap(r1, c1)];
                    qs_data_t src_10 = src[IdxMap(r1, c0)];
                    // for matrix[row, col], only in this case (row < col) is possible
                    qs_data_t src_01 = GetValue(src, r0, c1);
                    auto des_00 = m[0][0] * std::conj(m[0][0]) * src_00 + m[0][0] * std::conj(m[0][1]) * src_01
                                  + m[0][1] * std::conj(m[0][0]) * src_10 + m[0][1] * std::conj(m[0][1]) * src_11;
                    auto des_11 = m[1][0] * std::conj(m[1][0]) * src_00 + m[1][0] * std::conj(m[1][1]) * src_01
                                  + m[1][1] * std::conj(m[1][0]) * src_10 + m[1][1] * std::conj(m[1][1]) * src_11;
                    auto des_01 = m[0][0] * std::conj(m[1][0]) * src_00 + m[0][0] * std::conj(m[1][1]) * src_01
                                  + m[0][1] * std::conj(m[1][0]) * src_10 + m[0][1] * std::conj(m[1][1]) * src_11;
                    auto des_10 = m[1][0] * std::conj(m[0][0]) * src_00 + m[1][0] * std::conj(m[0][1]) * src_01
                                  + m[1][1] * std::conj(m[0][0]) * src_10 + m[1][1] * std::conj(m[0][1]) * src_11;

                    des[IdxMap(r0, c0)] = des_00;
                    des[IdxMap(r1, c1)] = des_11;
                    des[IdxMap(r1, c0)] = des_10;
                    SetValue(des, r0, c1, des_01);
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < (dim / 2); a++) {  // loop on the row
                auto r0 = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto r1 = r0 + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) {  // loop on the column
                    auto c0 = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    if (((r0 & mask.ctrl_mask) != mask.ctrl_mask)
                        && ((c0 & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                        continue;
                    }
                    auto c1 = c0 + mask.obj_mask;
                    qs_data_t src_00 = src[IdxMap(r0, c0)];
                    qs_data_t src_11 = src[IdxMap(r1, c1)];
                    qs_data_t src_10 = src[IdxMap(r1, c0)];
                    qs_data_t src_01 = GetValue(src, r0, c1);
                    qs_data_t des_00;
                    qs_data_t des_11;
                    qs_data_t des_01;
                    qs_data_t des_10;
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((c0 & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            des_00 = m[0][0] * std::conj(m[0][0]) * src_00 + m[0][0] * std::conj(m[0][1]) * src_01
                                     + m[0][1] * std::conj(m[0][0]) * src_10 + m[0][1] * std::conj(m[0][1]) * src_11;
                            des_11 = m[1][0] * std::conj(m[1][0]) * src_00 + m[1][0] * std::conj(m[1][1]) * src_01
                                     + m[1][1] * std::conj(m[1][0]) * src_10 + m[1][1] * std::conj(m[1][1]) * src_11;
                            des_01 = m[0][0] * std::conj(m[1][0]) * src_00 + m[0][0] * std::conj(m[1][1]) * src_01
                                     + m[0][1] * std::conj(m[1][0]) * src_10 + m[0][1] * std::conj(m[1][1]) * src_11;
                            des_10 = m[1][0] * std::conj(m[0][0]) * src_00 + m[1][0] * std::conj(m[0][1]) * src_01
                                     + m[1][1] * std::conj(m[0][0]) * src_10 + m[1][1] * std::conj(m[0][1]) * src_11;
                        } else {  // row in control but not column
                            des_00 = m[0][0] * src_00 + m[0][1] * src_10;
                            des_10 = m[1][0] * src_00 + m[1][1] * src_10;
                            des_01 = m[0][0] * src_01 + m[0][1] * src_11;
                            des_11 = m[1][0] * src_01 + m[1][1] * src_11;
                        }
                    } else {  // column in control but not row
                        des_00 = std::conj(m[0][0]) * src_00 + std::conj(m[0][1]) * src_01;
                        des_01 = std::conj(m[1][0]) * src_00 + std::conj(m[1][1]) * src_01;
                        des_10 = std::conj(m[0][0]) * src_10 + std::conj(m[0][1]) * src_11;
                        des_11 = std::conj(m[1][0]) * src_10 + std::conj(m[1][1]) * src_11;
                    }
                    des[IdxMap(r0, c0)] = des_00;
                    des[IdxMap(r1, c1)] = des_11;
                    des[IdxMap(r1, c0)] = des_10;
                    SetValue(des, r0, c1, des_01);
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des,
                                                                            const qbits_t& objs, const qbits_t& ctrls,
                                                                            const matrix_t& m, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        ApplyTwoQubitsMatrixNoCtrl(src, des, objs, ctrls, m, dim);
    } else {
        ApplyTwoQubitsMatrixCtrl(src, des, objs, ctrls, m, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrixNoCtrl(qs_data_p_t src, qs_data_p_t des,
                                                                                  const qbits_t& objs,
                                                                                  const qbits_t& ctrls,
                                                                                  const matrix_t& m, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    size_t mask1 = (1UL << objs[0]);
    size_t mask2 = (1UL << objs[1]);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask1;
            row[2] = row[0] + mask2;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask1;
                col[2] = col[0] + mask2;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        tmp_mat[i][j] = m[i][0] * GetValue(src, row[0], col[j])
                                        + m[i][1] * GetValue(src, row[1], col[j])
                                        + m[i][2] * GetValue(src, row[2], col[j])
                                        + m[i][3] * GetValue(src, row[3], col[j]);
                    }
                }

                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        auto new_value = tmp_mat[i][0] * std::conj(m[j][0]) + tmp_mat[i][1] * std::conj(m[j][1])
                                         + tmp_mat[i][2] * std::conj(m[j][2]) + tmp_mat[i][3] * std::conj(m[j][3]);
                        SetValue(des, row[i], col[j], new_value);
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTwoQubitsMatrixCtrl(qs_data_p_t src, qs_data_p_t des,
                                                                                const qbits_t& objs,
                                                                                const qbits_t& ctrls, const matrix_t& m,
                                                                                index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    size_t mask1 = (1UL << objs[0]);
    size_t mask2 = (1UL << objs[1]);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < (dim / 4); a++) {
            VT<index_t> row(4);  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                          row[0]);
            row[3] = row[0] + mask.obj_mask;
            row[1] = row[0] + mask1;
            row[2] = row[0] + mask2;
            for (index_t b = 0; b <= a; b++) {
                VT<index_t> col(4);  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              col[0]);
                if (((row[0] & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((col[0] & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                col[3] = col[0] + mask.obj_mask;
                col[1] = col[0] + mask1;
                col[2] = col[0] + mask2;
                VT<VT<qs_data_t>> tmp_mat(4, VT<qs_data_t>(4));
                if ((row[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // row in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = m[i][0] * GetValue(src, row[0], col[j])
                                            + m[i][1] * GetValue(src, row[1], col[j])
                                            + m[i][2] * GetValue(src, row[2], col[j])
                                            + m[i][3] * GetValue(src, row[3], col[j]);
                        }
                    }
                } else {  // row not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            tmp_mat[i][j] = GetValue(src, row[i], col[j]);
                        }
                    }
                }
                if ((col[0] & mask.ctrl_mask) == mask.ctrl_mask) {  // column in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            auto new_value = tmp_mat[i][0] * std::conj(m[j][0]) + tmp_mat[i][1] * std::conj(m[j][1])
                                             + tmp_mat[i][2] * std::conj(m[j][2]) + tmp_mat[i][3] * std::conj(m[j][3]);
                            SetValue(des, row[i], col[j], new_value);
                        }
                    }
                } else {  // column not in control
                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            SetValue(des, row[i], col[j], tmp_mat[i][j]);
                        }
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des,
                                                                       const qbits_t& objs, const qbits_t& ctrls,
                                                                       const matrix_t& m, index_t dim) {
    if (objs.size() == 1) {
        derived::ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        derived::ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for cpu backend.");
    }
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
