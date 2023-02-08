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
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

constexpr double m_sqrt1_2{0.707106781186547524400844362104849039};

namespace mindquantum::sim::densitymatrix::detail {
// Single qubit operator
// ========================================================================================================

// method is based on 'mq_vector' simulator, extended to densitymatrix
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                        const qbits_t& ctrls, const matrix_t& m, index_t dim) {
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
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    matrix_t m{{m_sqrt1_2, m_sqrt1_2}, {m_sqrt1_2, -m_sqrt1_2}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = -std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = -0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
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
