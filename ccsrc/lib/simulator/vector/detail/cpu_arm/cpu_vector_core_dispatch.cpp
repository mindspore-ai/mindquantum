/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <string_view>

#include "cpu_vector_arm_backend.h"

namespace mindquantum::sim::vector::detail {
namespace {
namespace baseline {
template <typename policy_t>
using base_policy_t = CPUVectorPolicyBase<policy_t, typename policy_t::calc_type>;

template <typename policy_t>
void ApplyMatrix(const typename policy_t::qs_data_p_t& src, typename policy_t::qs_data_p_t* des_p, const qbits_t& objs,
                 const qbits_t& ctrls, const arm_matrix_t<policy_t>& matrix, index_t dim) {
    if (objs.size() == 1) {
        base_policy_t<policy_t>::ApplySingleQubitMatrix(src, des_p, objs.front(), ctrls, matrix, dim);
        return;
    }
    if (objs.size() == 2) {
        base_policy_t<policy_t>::ApplyTwoQubitsMatrix(src, des_p, objs, ctrls, matrix, dim);
        return;
    }
    base_policy_t<policy_t>::ApplyNQubitsMatrix(src, des_p, objs, ctrls, matrix, dim);
}
}  // namespace baseline

template <typename policy_t>
[[nodiscard]] auto SelectedBackend() -> const ArmBackend<policy_t>& {
    static constexpr ArmBackend<policy_t> baseline_backend{"baseline", baseline::ApplyMatrix<policy_t>};
#if MQ_HAS_ARM_SVE_BACKEND
    static constexpr ArmBackend<policy_t> arm_sve_backend{"arm-sve", arm_sve::ApplyMatrix<policy_t>};
    static const auto* selected = cpu::HostHasArmSveFcma() ? &arm_sve_backend : &baseline_backend;
    return *selected;
#else
    return baseline_backend;
#endif
}

}  // namespace

template <typename derived_, typename calc_type_>
void CPUVectorPolicyArm<derived_, calc_type_>::ApplySingleQubitMatrix(
    const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit, const qbits_t& ctrls,
    const std::vector<std::vector<py_qs_data_t>>& matrix, index_t dim) {
    SelectedBackend<derived_>().apply_matrix(src, des_p, {obj_qubit}, ctrls, matrix, dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyArm<derived_, calc_type_>::ApplyTwoQubitsMatrix(
    const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs, const qbits_t& ctrls,
    const std::vector<std::vector<py_qs_data_t>>& matrix, index_t dim) {
    SelectedBackend<derived_>().apply_matrix(src, des_p, objs, ctrls, matrix, dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyArm<derived_, calc_type_>::ApplyNQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p,
                                                                  const qbits_t& objs, const qbits_t& ctrls,
                                                                  const std::vector<std::vector<py_qs_data_t>>& matrix,
                                                                  index_t dim) {
    SelectedBackend<derived_>().apply_matrix(src, des_p, objs, ctrls, matrix, dim);
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyArm<derived_, calc_type_>::GetBackendName() noexcept -> std::string_view {
    return SelectedBackend<derived_>().name;
}

template struct CPUVectorPolicyArm<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyArm<CPUVectorPolicyArmDouble, double>;
}  // namespace mindquantum::sim::vector::detail
