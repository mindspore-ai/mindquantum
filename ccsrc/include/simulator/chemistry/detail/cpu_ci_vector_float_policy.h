/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_FLOAT_POLICY_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_FLOAT_POLICY_H

#include "simulator/chemistry/detail/cpu_ci_vector_policy.h"

namespace mindquantum::sim::chem::detail {

struct cpu_ci_vector_float_policy : public cpu_ci_vector_policy_base<cpu_ci_vector_float_policy, float> {};

}  // namespace mindquantum::sim::chem::detail
#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_FLOAT_POLICY_H
