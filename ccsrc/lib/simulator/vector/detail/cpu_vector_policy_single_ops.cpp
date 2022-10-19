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
#include "simulator/vector/detail/cpu_vector_policy.hpp"

#ifndef M_SQRT1_2
#    define M_SQRT1_2 1.12837916709551257390
#endif  // !M_SQRT1_2

namespace mindquantum::sim::vector::detail {
// Single qubit operator
// ========================================================================================================
void CPUVectorPolicyBase::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                               const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& gate,
                                               index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP_FOR(dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto m = i + mask.obj_mask;
                auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                des[i] = v00;
                des[j] = v01;
                des[k] = v10;
                des[m] = v11;
            })
        // clang-format on
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto m = i + mask.obj_mask;
                    auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                    auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                    auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                    auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                    des[i] = v00;
                    des[j] = v01;
                    des[k] = v10;
                    des[m] = v11;
                }
            })
    }
}
void CPUVectorPolicyBase::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                 const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& m,
                                                 index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
#ifdef INTRIN
    gate_matrix_t gate = {{m[0][0], m[0][1]}, {m[1][0], m[1][1]}};
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m256d mm[2];
    __m256d mmt[2];
    INTRIN_gene_2d_mm_and_mmt(gate, mm, mmt, neg);
#endif
    if (!mask.ctrl_mask) {
#ifdef INTRIN
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                __m256d mul_res;
                INTRIN_M2_dot_V2(src, i, j, mm, mmt, mul_res);
                INTRIN_m256_to_host2(mul_res, des + i, des + j);
            })
#else
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                des[i] = t1;
                des[j] = t2;
            })
#endif
    } else {
#ifdef INTRIN
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    __m256d mul_res;
                    INTRIN_M2_dot_V2(src, i, j, mm, mmt, mul_res);
                    INTRIN_m256_to_host2(mul_res, des + i, des + j);
                }
            });
#else
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                    auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                    des[i] = t1;
                    des[j] = t2;
                }
            });
#endif
    }
}

void CPUVectorPolicyBase::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

void CPUVectorPolicyBase::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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

void CPUVectorPolicyBase::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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

void CPUVectorPolicyBase::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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
}  // namespace mindquantum::sim::vector::detail
