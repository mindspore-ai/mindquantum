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
// Z like operator
// ========================================================================================================

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyZLike(qs_data_p_t qs, const qbits_t& objs,
                                                                  const qbits_t& ctrls, qs_data_t val, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t k = 0; k < (dim / 2); k++) {  // loop on the row
                auto r0 = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                auto r1 = r0 | mask.obj_mask;
                for (index_t l = 0; l < k; l++) {  // loop on the column
                    auto c0 = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto c1 = c0 | mask.obj_mask;
                    qs[IdxMap(r1, c1)] *= std::norm(val);
                    qs[IdxMap(r1, c0)] *= val;
                    SelfMultiply(qs, r0, c1, std::conj(val));
                }
                // diagonal case
                qs[IdxMap(r1, r1)] *= std::norm(val);
                qs[IdxMap(r1, r0)] *= val;
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
                            qs[IdxMap(r1, c1)] *= std::norm(val);
                            qs[IdxMap(r1, c0)] *= val;
                            SelfMultiply(qs, r0, c1, std::conj(val));
                        } else {  // row in control but not column
                            qs[IdxMap(r1, c1)] *= val;
                            qs[IdxMap(r1, c0)] *= val;
                        }
                    } else {  // column in control but not row
                        qs[IdxMap(r1, c1)] *= std::conj(val);
                        SelfMultiply(qs, r0, c1, std::conj(val));
                    }
                }
                // diagonal case
                if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                    qs[IdxMap(r1, r0)] *= val;
                    qs[IdxMap(r1, r1)] *= std::norm(val);
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                              index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, -1, dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySGate(qs_data_p_t qs, const qbits_t& objs,
                                                                  const qbits_t& ctrls, index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(0, 1), dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplySdag(qs_data_p_t qs, const qbits_t& objs,
                                                                 const qbits_t& ctrls, index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(0, -1), dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                              index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(1, 1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTdag(qs_data_p_t qs, const qbits_t& objs,
                                                                 const qbits_t& ctrls, index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(1, -1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyPS(qs_data_p_t qs, const qbits_t& objs,
                                                               const qbits_t& ctrls, calc_type val, index_t dim,
                                                               bool diff) {
    if (!diff) {
        derived::ApplyZLike(qs, objs, ctrls, qs_data_t(std::cos(val), std::sin(val)), dim);
    } else {
        SingleQubitGateMask mask(objs, ctrls);
        auto e = -std::sin(val) + IMAGE_I * std::cos(val);
        if (!mask.ctrl_mask) {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t k = 0; k < (dim / 2); k++) {  // loop on the row
                    auto r0 = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                    auto r1 = r0 | mask.obj_mask;
                    for (index_t l = 0; l <= k; l++) {  // loop on the column
                        auto c0 = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                        auto c1 = c0 | mask.obj_mask;
                        qs[IdxMap(r0, c0)] = 0;
                        qs[IdxMap(r0, c1)] = 0;
                        qs[IdxMap(r1, c0)] = 0;
                    }
                })
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t k = 0; k < (dim / 2); k++) {  // loop on the row
                    auto r0 = ((k & mask.obj_high_mask) << 1) + (k & mask.obj_low_mask);
                    auto r1 = r0 | mask.obj_mask;
                    for (index_t l = 0; l <= k; l++) {  // loop on the column
                        auto c0 = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);

                        if (((r0 & mask.ctrl_mask) != mask.ctrl_mask)
                            && ((c0 & mask.ctrl_mask) != mask.ctrl_mask)) {  // both not in control
                            continue;
                        }
                        auto c1 = c0 | mask.obj_mask;
                        if ((r0 & mask.ctrl_mask) == mask.ctrl_mask) {
                            if ((c0 & mask.ctrl_mask) == mask.ctrl_mask) {  // both in control
                                qs[IdxMap(r0, c0)] = 0;
                                qs[IdxMap(r0, c1)] = 0;
                                qs[IdxMap(r1, c0)] = 0;
                            } else {  // only row in control
                                qs[IdxMap(r0, c0)] = 0;
                                qs[IdxMap(r0, c1)] = 0;
                                qs[IdxMap(r1, c0)] *= e;
                                qs[IdxMap(r1, c1)] *= e;
                            }
                        } else {  // only column on control
                            qs[IdxMap(r0, c0)] = 0;
                            qs[IdxMap(r0, c1)] *= std::conj(e);
                            qs[IdxMap(r1, c0)] = 0;
                            qs[IdxMap(r1, c1)] *= std::conj(e);
                        }
                    }
                })
            derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
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
