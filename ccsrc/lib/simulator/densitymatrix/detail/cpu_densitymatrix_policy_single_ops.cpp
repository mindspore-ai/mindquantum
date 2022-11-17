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

constexpr double m_sqrt1_2{0.707106781186547524400844362104849039};

namespace mindquantum::sim::densitymatrix::detail {
// Single qubit operator
// ========================================================================================================

// method is based on 'mq_vector' simulator, extended to densitymatrix
void CPUDensityMatrixPolicyBase::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                        const qbits_t& ctrls,
                                                        const matrix_t& m, index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) { // loop on the row
                auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) { // loop on the column
                    auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    auto q = p + mask.obj_mask;
                    qs_data_t src_ip{src[IdxMap(i, p)]};
                    qs_data_t src_jq{src[IdxMap(j, q)]};
                    qs_data_t src_jp{src[IdxMap(j, p)]};
                    qs_data_t src_iq;
                    if (i > q) {  // for matrix[row, col], only in this case (row < col) is possible
                        src_iq = src[IdxMap(i, q)];
                    } else {
                        src_iq = std::conj(src[IdxMap(q, i)]);
                    }
                    auto des_ip = m[0][0] * std::conj(m[0][0]) * src_ip + m[0][0] * std::conj(m[0][1]) * src_iq
                                  + m[0][1] * std::conj(m[0][0]) * src_jp + m[0][1] * std::conj(m[0][1]) * src_jq;
                    auto des_jq = m[1][0] * std::conj(m[1][0]) * src_ip + m[1][0] * std::conj(m[1][1]) * src_iq
                                  + m[1][1] * std::conj(m[1][0]) * src_jp + m[1][1] * std::conj(m[1][1]) * src_jq;
                    auto des_iq = m[0][0] * std::conj(m[1][0]) * src_ip + m[0][0] * std::conj(m[1][1]) * src_iq
                                  + m[0][1] * std::conj(m[1][0]) * src_jp + m[0][1] * std::conj(m[1][1]) * src_jq;
                    auto des_jp = m[1][0] * std::conj(m[0][0]) * src_ip + m[1][0] * std::conj(m[0][1]) * src_iq
                                  + m[1][1] * std::conj(m[0][0]) * src_jp + m[1][1] * std::conj(m[0][1]) * src_jq;

                    des[IdxMap(i, p)] = des_ip;
                    des[IdxMap(j, q)] = des_jq;
                    des[IdxMap(j, p)] = des_jp;
                    if (i > q) {
                        des[IdxMap(i, q)] = des_iq;
                    } else {
                        des[IdxMap(q, i)] = std::conj(des_iq);
                    }
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) { // loop on the row
                auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                for (index_t b = 0; b <= a; b++) { // loop on the column
                    auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    if (((i & mask.ctrl_mask) != mask.ctrl_mask)
                        && ((p & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                        continue;
                    }
                    auto q = p + mask.obj_mask;
                    qs_data_t src_ip{src[IdxMap(i, p)]};
                    qs_data_t src_jq{src[IdxMap(j, q)]};
                    qs_data_t src_jp{src[IdxMap(j, p)]};
                    qs_data_t src_iq;
                    if (i > q) {  // for qs[row, col], only in this case that (row < col) is possible
                        src_iq = src[IdxMap(i, q)];
                    } else {
                        src_iq = std::conj(src[IdxMap(q, i)]);
                    }
                    qs_data_t des_ip;
                    qs_data_t des_jq;
                    qs_data_t des_iq;
                    qs_data_t des_jp;
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((p & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            des_ip = m[0][0] * std::conj(m[0][0]) * src_ip + m[0][0] * std::conj(m[0][1]) * src_iq
                                     + m[0][1] * std::conj(m[0][0]) * src_jp + m[0][1] * std::conj(m[0][1]) * src_jq;
                            des_jq = m[1][0] * std::conj(m[1][0]) * src_ip + m[1][0] * std::conj(m[1][1]) * src_iq
                                     + m[1][1] * std::conj(m[1][0]) * src_jp + m[1][1] * std::conj(m[1][1]) * src_jq;
                            des_iq = m[0][0] * std::conj(m[1][0]) * src_ip + m[0][0] * std::conj(m[1][1]) * src_iq
                                     + m[0][1] * std::conj(m[1][0]) * src_jp + m[0][1] * std::conj(m[1][1]) * src_jq;
                            des_jp = m[1][0] * std::conj(m[0][0]) * src_ip + m[1][0] * std::conj(m[0][1]) * src_iq
                                     + m[1][1] * std::conj(m[0][0]) * src_jp + m[1][1] * std::conj(m[0][1]) * src_jq;
                        } else {  // row in control but not column
                            des_ip = m[0][0] * src_ip + m[0][1] * src_jp;
                            des_jp = m[1][0] * src_ip + m[1][1] * src_jp;
                            des_iq = m[0][0] * src_iq + m[0][1] * src_jq;
                            des_jq = m[1][0] * src_iq + m[1][1] * src_jq;
                        }
                    } else {  // column in control but not row
                        des_ip = std::conj(m[0][0]) * src_ip + std::conj(m[0][1]) * src_iq;
                        des_iq = std::conj(m[1][0]) * src_ip + std::conj(m[1][1]) * src_iq;
                        des_jp = std::conj(m[0][0]) * src_jp + std::conj(m[0][1]) * src_jq;
                        des_jq = std::conj(m[1][0]) * src_jp + std::conj(m[1][1]) * src_jq;
                    }
                    des[IdxMap(i, p)] = des_ip;
                    des[IdxMap(j, q)] = des_jq;
                    des[IdxMap(j, p)] = des_jp;
                    if (i > q) {
                        des[IdxMap(i, q)] = des_iq;
                    } else {
                        des[IdxMap(q, i)] = std::conj(des_iq);
                    }
                }
            })
    }
}

void CPUDensityMatrixPolicyBase::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    matrix_t m{{m_sqrt1_2, m_sqrt1_2}, {m_sqrt1_2, -m_sqrt1_2}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

void CPUDensityMatrixPolicyBase::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = -std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = -0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

void CPUDensityMatrixPolicyBase::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

void CPUDensityMatrixPolicyBase::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                         index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    matrix_t m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}
}  // namespace mindquantum::sim::densitymatrix::detail
