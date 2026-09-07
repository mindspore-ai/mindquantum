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

#include <vector>

#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/circuit_tiling/executor.h"
#include "simulator/vector/detail/cpu_vector_policy.h"

namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyCircuitTile(qs_data_p_t* qs_p,
                                                                 const circuit_tiling::PreparedTile<py_qs_data_t>& tile,
                                                                 index_t dim) {
    auto& qs = *qs_p;
    if (qs == nullptr) {
        qs = derived::InitState(dim);
    }
    circuit_tiling::ApplyTile(qs, tile, omp::TeamSize(dim));
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyAdjointTileProgram(
    qs_data_p_t* shared_state, const std::vector<qs_data_p_t*>& batch_states,
    const circuit_tiling::AdjointProgram<py_qs_data_t>& program, index_t dim, VVT<py_qs_data_t>* output,
    bool conjugate_output) {
    if (*shared_state == nullptr) {
        *shared_state = derived::InitState(dim);
    }
    std::vector<qs_data_p_t> states;
    states.reserve(batch_states.size());
    for (auto* state : batch_states) {
        if (*state == nullptr) {
            *state = derived::InitState(dim);
        }
        states.push_back(*state);
    }
    circuit_tiling::ApplyAdjointProgram(*shared_state, states, program, output, conjugate_output, omp::TeamSize(dim));
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
