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
// X like operator
// ========================================================================================================

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyXLike(qs_data_p_t qs, const qbits_t& objs,
                                                                  const qbits_t& ctrls, qs_data_t v1, qs_data_t v2,
                                                                  index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t k = 0; k < (dim / 2); k++) {  // loop on the row
                auto r0 = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                auto r1 = r0 | mask.obj_mask;
                for (index_t l = 0; l < k; l++) {  // loop on the column
                    auto c0 = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto c1 = c0 | mask.obj_mask;
                    auto tmp = qs[IdxMap(r0, c0)];
                    qs[IdxMap(r0, c0)] = qs[IdxMap(r1, c1)] * std::norm(v1);
                    qs[IdxMap(r1, c1)] = tmp * std::norm(v2);
                    tmp = qs[IdxMap(r1, c0)];
                    qs[IdxMap(r1, c0)] = GetValue(qs, r0, c1) * v2 * std::conj(v1);
                    // for matrix[row, col], only in this case that (row < col) is possible
                    SetValue(qs, r0, c1, tmp * v1 * std::conj(v2));
                }
                // diagonal case
                auto tmp = qs[IdxMap(r0, r0)];
                qs[IdxMap(r0, r0)] = qs[IdxMap(r1, r1)] * std::norm(v1);
                qs[IdxMap(r1, r1)] = tmp * std::norm(v2);
                qs[IdxMap(r1, r0)] = std::conj(qs[IdxMap(r1, r0)]) * v2 * std::conj(v1);
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t k = 0; k < (dim / 2); k++) {  // loop on the row
                auto r0 = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                auto r1 = r0 | mask.obj_mask;
                for (index_t l = 0; l < k; l++) {  // loop on the column
                    auto c0 = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    if (((r0 & mask.ctrl_mask) != mask.ctrl_mask)
                        && ((c0 & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                        continue;
                    }
                    auto c1 = c0 | mask.obj_mask;
                    if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                        if ((c0 & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                            auto tmp = qs[IdxMap(r0, c0)];
                            qs[IdxMap(r0, c0)] = qs[IdxMap(r1, c1)] * std::norm(v1);
                            qs[IdxMap(r1, c1)] = tmp * std::norm(v2);
                            tmp = qs[IdxMap(r1, c0)];
                            qs[IdxMap(r1, c0)] = GetValue(qs, r0, c1) * v2 * std::conj(v1);
                            SetValue(qs, r0, c1, tmp * v1 * std::conj(v2));
                        } else {  // row in control but not column
                            auto tmp = qs[IdxMap(r0, c0)];
                            qs[IdxMap(r0, c0)] = qs[IdxMap(r1, c0)] * v1;
                            qs[IdxMap(r1, c0)] = tmp * v2;
                            tmp = GetValue(qs, r0, c1);
                            SetValue(qs, r0, c1, qs[IdxMap(r1, c1)] * v1);
                            qs[IdxMap(r1, c1)] = tmp * v2;
                        }
                    } else {  // column in control but not row
                        auto tmp = qs[IdxMap(r1, c0)];
                        qs[IdxMap(r1, c0)] = qs[IdxMap(r1, c1)] * std::conj(v1);
                        qs[IdxMap(r1, c1)] = tmp * std::conj(v2);
                        tmp = qs[IdxMap(r0, c0)];
                        qs[IdxMap(r0, c0)] = GetValue(qs, r0, c1) * std::conj(v1);
                        SetValue(qs, r0, c1, tmp * std::conj(v2));
                    }
                }
                // diagonal case
                if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto tmp = qs[IdxMap(r0, r0)];
                    qs[IdxMap(r0, r0)] = qs[IdxMap(r1, r1)] * std::norm(v1);
                    qs[IdxMap(r1, r1)] = tmp * std::norm(v2);
                    qs[IdxMap(r1, r0)] = std::conj(qs[IdxMap(r1, r0)]) * v2 * std::conj(v1);
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                              index_t dim) {
    derived::ApplyXLike(qs, objs, ctrls, 1, 1, dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                              index_t dim) {
    derived::ApplyXLike(qs, objs, ctrls, IMAGE_MI, IMAGE_I, dim);
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
