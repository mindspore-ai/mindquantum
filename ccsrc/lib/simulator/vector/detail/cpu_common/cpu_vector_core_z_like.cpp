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

#include "math/pr/parameter_resolver.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.hpp"
#endif
#include "simulator/vector/detail/cpu_vector_policy.hpp"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyZLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                           qs_data_t val, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                qs[i] *= val;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    qs[i] *= val;
                }
            })
    }
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                       index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, -1, dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySGate(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                           index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(0, 1), dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplySdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                          index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(0, -1), dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                       index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(1, 1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyTdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                          index_t dim) {
    derived::ApplyZLike(qs, objs, ctrls, qs_data_t(1, -1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyPS(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    if (!diff) {
        derived::ApplyZLike(qs, objs, ctrls, qs_data_t(std::cos(val), std::sin(val)), dim);
    } else {
        SingleQubitGateMask mask(objs, ctrls);
        auto e = -std::sin(val) + IMAGE_I * std::cos(val);
        if (!mask.ctrl_mask) {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs[i] = 0;
                    qs[j] *= e;
                })
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs[i] = 0;
                        qs[j] *= e;
                    }
                })
            derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
