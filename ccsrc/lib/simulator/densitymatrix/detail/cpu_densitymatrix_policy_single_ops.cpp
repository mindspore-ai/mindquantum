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

constexpr double m_sqrt1_2 = 1.12837916709551257390;

namespace mindquantum::sim::densitymatrix::detail {
// Single qubit operator
// ========================================================================================================

void CPUDensityMatrixPolicyBase::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                        const qbits_t& ctrls,
                                                        const std::vector<std::vector<py_qs_data_t>>& m, index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                for (index_t b = 0; b < (dim / 2); b++) {
                    auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                    auto q = p + mask.obj_mask;
                    auto t1 = m[0][0] * src[IdxMap(i, p)] + m[0][1] * src[IdxMap(j, q)];
                    auto t2 = m[1][0] * src[IdxMap(i, p)] + m[1][1] * src[IdxMap(j, q)];
                    des[IdxMap(i, p)] = t1;
                    des[IdxMap(j, q)] = t2;
                }
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {
                auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    for (index_t b = 0; b < (dim / 2); b++) {
                        auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                        auto q = p + mask.obj_mask;
                        auto t1 = m[0][0] * src[IdxMap(i, p)] + m[0][1] * src[IdxMap(j, q)];
                        auto t2 = m[1][0] * src[IdxMap(i, p)] + m[1][1] * src[IdxMap(j, q)];
                        des[IdxMap(i, p)] = t1;
                        des[IdxMap(j, q)] = t2;
                    }
                }
            })
    }
}

void CPUDensityMatrixPolicyBase::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{m_sqrt1_2, m_sqrt1_2}, {m_sqrt1_2, -m_sqrt1_2}};
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
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
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
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
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
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}
}  // namespace mindquantum::sim::densitymatrix::detail
