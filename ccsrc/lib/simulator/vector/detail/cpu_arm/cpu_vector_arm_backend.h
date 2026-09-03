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

#ifndef LIB_SIMULATOR_VECTOR_DETAIL_CPU_ARM_CPU_VECTOR_ARM_BACKEND_H_
#define LIB_SIMULATOR_VECTOR_DETAIL_CPU_ARM_CPU_VECTOR_ARM_BACKEND_H_

#include <vector>

#include "simulator/cpu_features.h"
#include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#include "simulator/vector/detail/cpu_vector_arm_float_policy.h"

namespace mindquantum::sim::vector::detail {
template <typename policy_t>
using arm_matrix_t = std::vector<std::vector<typename policy_t::py_qs_data_t>>;

template <typename policy_t>
struct ArmBackend {
    using apply_matrix_t = void (*)(const typename policy_t::qs_data_p_t&, typename policy_t::qs_data_p_t*,
                                    const qbits_t&, const qbits_t&, const arm_matrix_t<policy_t>&, index_t);

    const char* name;
    apply_matrix_t apply_matrix;
};

namespace arm_sve {
template <typename policy_t>
void ApplyMatrix(const typename policy_t::qs_data_p_t& src, typename policy_t::qs_data_p_t* des_p, const qbits_t& objs,
                 const qbits_t& ctrls, const arm_matrix_t<policy_t>& matrix, index_t dim);
}  // namespace arm_sve
}  // namespace mindquantum::sim::vector::detail

#endif  // LIB_SIMULATOR_VECTOR_DETAIL_CPU_ARM_CPU_VECTOR_ARM_BACKEND_H_
