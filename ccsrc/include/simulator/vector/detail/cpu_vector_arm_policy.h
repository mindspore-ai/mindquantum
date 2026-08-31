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

#ifndef INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_ARM_POLICY_H_
#define INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_ARM_POLICY_H_

#include <string_view>
#include <vector>

#include "simulator/vector/detail/cpu_vector_policy.h"

namespace mindquantum::sim::vector::detail {
// The Arm policies share one ISA-dispatched dense matrix implementation.
// Precision is a compile-time property, while the usable instruction set is
// selected once at runtime for each concrete policy.  Circuit tiles are
// executed by the shared, runtime-dispatched circuit-tiling engine.
template <typename derived_, typename calc_type_>
struct CPUVectorPolicyArm : public CPUVectorPolicyBase<derived_, calc_type_> {
    using base_t = CPUVectorPolicyBase<derived_, calc_type_>;
    using typename base_t::py_qs_data_t;
    using typename base_t::qs_data_p_t;

    static void ApplySingleQubitMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, qbit_t obj_qubit,
                                       const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& matrix,
                                       index_t dim);
    static void ApplyTwoQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                     const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& matrix,
                                     index_t dim);
    static void ApplyNQubitsMatrix(const qs_data_p_t& src, qs_data_p_t* des_p, const qbits_t& objs,
                                   const qbits_t& ctrls, const std::vector<std::vector<py_qs_data_t>>& matrix,
                                   index_t dim);
    [[nodiscard]] static auto GetBackendName() noexcept -> std::string_view;
};
}  // namespace mindquantum::sim::vector::detail

#endif  // INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_ARM_POLICY_H_
