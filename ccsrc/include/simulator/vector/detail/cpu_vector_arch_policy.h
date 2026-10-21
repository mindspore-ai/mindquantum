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

#ifndef INCLUDE_VECTOR_DETAIL_CPU_VECTOR_ARCH_POLICY_HPP
#define INCLUDE_VECTOR_DETAIL_CPU_VECTOR_ARCH_POLICY_HPP

// The CPU state-vector policies compiled into mqsim_vector_cpu.
//
// CMake selects one backend directory per target architecture and advertises that choice through exactly one
// MQ_VECTOR_ARCH_* definition on the mqsim_vector_cpu target (ccsrc/lib/simulator/vector/detail/CMakeLists.txt). This
// header is the only place that maps the choice onto concrete policy types; everything else names the aliases below.

#if defined(MQ_VECTOR_ARCH_X86_64)
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"

namespace mindquantum::sim::vector::detail {
using CPUVectorPolicyFloat = CPUVectorPolicyAvxFloat;
using CPUVectorPolicyDouble = CPUVectorPolicyAvxDouble;
}  // namespace mindquantum::sim::vector::detail

#elif defined(MQ_VECTOR_ARCH_ARM64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"

namespace mindquantum::sim::vector::detail {
using CPUVectorPolicyFloat = CPUVectorPolicyArmFloat;
using CPUVectorPolicyDouble = CPUVectorPolicyArmDouble;
}  // namespace mindquantum::sim::vector::detail

#else
#    error "No CPU state-vector backend selected (CMake defines MQ_VECTOR_ARCH_X86_64 or MQ_VECTOR_ARCH_ARM64)"
#endif

#endif  // INCLUDE_VECTOR_DETAIL_CPU_VECTOR_ARCH_POLICY_HPP
