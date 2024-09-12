/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "config/openmp.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/utils.h"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.h"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.h"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySWAP(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                 const qbits_t& ctrls, index_t dim) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
                index_t r0;  // row index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                              r0);
                auto r3 = r0 + mask.obj_mask;
                auto r1 = r0 + mask.obj_min_mask;
                auto r2 = r0 + mask.obj_max_mask;
                for (index_t b = 0; b < a; b++) {
                    index_t c0;  // column index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, c0);
                    auto c3 = c0 + mask.obj_mask;
                    auto c1 = c0 + mask.obj_min_mask;
                    auto c2 = c0 + mask.obj_max_mask;
                    SwapValue(qs, r0, c1, r0, c2, 1);
                    SwapValue(qs, r3, c1, r3, c2, 1);
                    SwapValue(qs, r1, c0, r2, c0, 1);
                    SwapValue(qs, r1, c3, r2, c3, 1);
                    SwapValue(qs, r1, c1, r2, c2, 1);
                    SwapValue(qs, r1, c2, r2, c1, 1);
                }
                // diagonal case
                qs_data_t tmp;
                tmp = qs[IdxMap(r3, r1)];
                qs[IdxMap(r3, r1)] = qs[IdxMap(r3, r2)];
                qs[IdxMap(r3, r2)] = tmp;

                tmp = qs[IdxMap(r1, r0)];
                qs[IdxMap(r1, r0)] = qs[IdxMap(r2, r0)];
                qs[IdxMap(r2, r0)] = tmp;

                tmp = qs[IdxMap(r1, r1)];
                qs[IdxMap(r1, r1)] = qs[IdxMap(r2, r2)];
                qs[IdxMap(r2, r2)] = tmp;

                SetValue(qs, r2, r1, std::conj(GetValue(qs, r2, r1)));
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
                index_t r0;  // row index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a,
                              r0);
                auto r3 = r0 + mask.obj_mask;
                auto r1 = r0 + mask.obj_min_mask;
                auto r2 = r0 + mask.obj_max_mask;
                for (index_t b = 0; b < a; b++) {
                    index_t c0;  // column index of reduced matrix entry
                    SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                                  b, c0);
                    if (((r0 & mask.ctrl_mask) != mask.ctrl_mask)
                        && ((c0 & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                        continue;
                    }
                    auto c3 = c0 + mask.obj_mask;
                    auto c1 = c0 + mask.obj_min_mask;
                    auto c2 = c0 + mask.obj_max_mask;
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((c0 & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            SwapValue(qs, r0, c1, r0, c2, 1);
                            SwapValue(qs, r3, c1, r3, c2, 1);
                            SwapValue(qs, r1, c0, r2, c0, 1);
                            SwapValue(qs, r1, c3, r2, c3, 1);
                            SwapValue(qs, r1, c1, r2, c2, 1);
                            SwapValue(qs, r1, c2, r2, c1, 1);
                        } else {  // only row in control
                            SwapValue(qs, r1, c0, r2, c0, 1);
                            SwapValue(qs, r1, c1, r2, c1, 1);
                            SwapValue(qs, r1, c2, r2, c2, 1);
                            SwapValue(qs, r1, c3, r2, c3, 1);
                        }
                    } else {  // only column in control
                        SwapValue(qs, r0, c1, r0, c2, 1);
                        SwapValue(qs, r1, c1, r1, c2, 1);
                        SwapValue(qs, r2, c1, r2, c2, 1);
                        SwapValue(qs, r3, c1, r3, c2, 1);
                    }
                }
                // diagonal case
                if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                    qs_data_t tmp;
                    tmp = qs[IdxMap(r3, r1)];
                    qs[IdxMap(r3, r1)] = qs[IdxMap(r3, r2)];
                    qs[IdxMap(r3, r2)] = tmp;

                    tmp = qs[IdxMap(r1, r0)];
                    qs[IdxMap(r1, r0)] = qs[IdxMap(r2, r0)];
                    qs[IdxMap(r2, r0)] = tmp;

                    tmp = qs[IdxMap(r1, r1)];
                    qs[IdxMap(r1, r1)] = qs[IdxMap(r2, r2)];
                    qs[IdxMap(r2, r2)] = tmp;

                    SetValue(qs, r2, r1, std::conj(GetValue(qs, r2, r1)));
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyISWAP(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                  const qbits_t& ctrls, bool daggered, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        ApplyISWAPNoCtrl(qs_p, objs, ctrls, daggered, dim);
    } else {
        ApplyISWAPCtrl(qs_p, objs, ctrls, daggered, dim);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyISWAPNoCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                        const qbits_t& ctrls, bool daggered,
                                                                        index_t dim) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            index_t r0;  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a, r0);
            auto r3 = r0 + mask.obj_mask;
            auto r1 = r0 + mask.obj_min_mask;
            auto r2 = r0 + mask.obj_max_mask;
            for (index_t b = 0; b < a; b++) {
                index_t c0;  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              c0);
                auto c3 = c0 + mask.obj_mask;
                auto c1 = c0 + mask.obj_min_mask;
                auto c2 = c0 + mask.obj_max_mask;
                SwapValue(qs, r0, c1, r0, c2, frac * IMAGE_MI);
                SwapValue(qs, r3, c1, r3, c2, frac * IMAGE_MI);
                SwapValue(qs, r1, c0, r2, c0, frac * IMAGE_I);
                SwapValue(qs, r1, c3, r2, c3, frac * IMAGE_I);
                SwapValue(qs, r1, c1, r2, c2, 1);
                SwapValue(qs, r1, c2, r2, c1, 1);
            }
            // diagonal case
            qs_data_t tmp;
            tmp = qs[IdxMap(r3, r1)];
            qs[IdxMap(r3, r1)] = frac * IMAGE_MI * qs[IdxMap(r3, r2)];
            qs[IdxMap(r3, r2)] = frac * IMAGE_MI * tmp;

            tmp = qs[IdxMap(r1, r0)];
            qs[IdxMap(r1, r0)] = frac * IMAGE_I * qs[IdxMap(r2, r0)];
            qs[IdxMap(r2, r0)] = frac * IMAGE_I * tmp;

            tmp = qs[IdxMap(r1, r1)];
            qs[IdxMap(r1, r1)] = qs[IdxMap(r2, r2)];
            qs[IdxMap(r2, r2)] = tmp;

            SetValue(qs, r2, r1, std::conj(GetValue(qs, r2, r1)));
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyISWAPCtrl(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, bool daggered,
                                                                      index_t dim) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    DoubleQubitGateMask mask(objs, ctrls);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t a = 0; a < static_cast<omp::idx_t>(dim / 4); a++) {
            index_t r0;  // row index of reduced matrix entry
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, a, r0);
            auto r3 = r0 + mask.obj_mask;
            auto r1 = r0 + mask.obj_min_mask;
            auto r2 = r0 + mask.obj_max_mask;
            for (index_t b = 0; b < a; b++) {
                index_t c0;  // column index of reduced matrix entry
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, b,
                              c0);
                if (((r0 & mask.ctrl_mask) != mask.ctrl_mask)
                    && ((c0 & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                    continue;
                }
                auto c3 = c0 + mask.obj_mask;
                auto c1 = c0 + mask.obj_min_mask;
                auto c2 = c0 + mask.obj_max_mask;
                if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                    if ((c0 & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                        SwapValue(qs, r0, c1, r0, c2, frac * IMAGE_MI);
                        SwapValue(qs, r3, c1, r3, c2, frac * IMAGE_MI);
                        SwapValue(qs, r1, c0, r2, c0, frac * IMAGE_I);
                        SwapValue(qs, r1, c3, r2, c3, frac * IMAGE_I);
                        SwapValue(qs, r1, c1, r2, c2, 1);
                        SwapValue(qs, r1, c2, r2, c1, 1);
                    } else {  // only row in control
                        SwapValue(qs, r1, c0, r2, c0, frac * IMAGE_I);
                        SwapValue(qs, r1, c1, r2, c1, frac * IMAGE_I);
                        SwapValue(qs, r1, c2, r2, c2, frac * IMAGE_I);
                        SwapValue(qs, r1, c3, r2, c3, frac * IMAGE_I);
                    }
                } else {  // only column in control
                    SwapValue(qs, r0, c1, r0, c2, frac * IMAGE_MI);
                    SwapValue(qs, r1, c1, r1, c2, frac * IMAGE_MI);
                    SwapValue(qs, r2, c1, r2, c2, frac * IMAGE_MI);
                    SwapValue(qs, r3, c1, r3, c2, frac * IMAGE_MI);
                }
            }
            // diagonal case
            if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                qs_data_t tmp;
                tmp = qs[IdxMap(r3, r1)];
                qs[IdxMap(r3, r1)] = frac * IMAGE_MI * qs[IdxMap(r3, r2)];
                qs[IdxMap(r3, r2)] = frac * IMAGE_MI * tmp;

                tmp = qs[IdxMap(r1, r0)];
                qs[IdxMap(r1, r0)] = frac * IMAGE_I * qs[IdxMap(r2, r0)];
                qs[IdxMap(r2, r0)] = frac * IMAGE_I * tmp;

                tmp = qs[IdxMap(r1, r1)];
                qs[IdxMap(r1, r1)] = qs[IdxMap(r2, r2)];
                qs[IdxMap(r2, r2)] = tmp;

                SetValue(qs, r2, r1, std::conj(GetValue(qs, r2, r1)));
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySWAPalpha(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                      const qbits_t& ctrls, calc_type val, index_t dim,
                                                                      bool diff) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto e = std::exp(IMAGE_I * static_cast<calc_type_>(M_PI) * val);
    auto a = (static_cast<calc_type_>(1) + e) / static_cast<calc_type_>(2);
    auto b = (static_cast<calc_type_>(1) - e) / static_cast<calc_type_>(2);
    matrix_t m = {{1, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, 1}};
    if (diff) {
        a = IMAGE_I * static_cast<calc_type_>(M_PI_2) * e;
        b = IMAGE_MI * static_cast<calc_type_>(M_PI_2) * e;
        m = {{0, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, 0}};
    }
    derived::ApplyTwoQubitsMatrix(qs, qs_p, objs, ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyGivens(qs_data_p_t* qs_p, const qbits_t& objs,
                                                                   const qbits_t& ctrls, calc_type val, index_t dim,
                                                                   bool diff) {
    auto& qs = (*qs_p);
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val);
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val);
    }
    matrix_t m = {{1, 0, 0, 0}, {0, c, -s, 0}, {0, s, c, 0}, {0, 0, 0, 1}};
    derived::ApplyTwoQubitsMatrix(qs, qs_p, objs, ctrls, m, dim);
}
#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
