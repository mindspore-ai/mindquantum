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
#ifndef INCLUDE_VECTOR_DETAIL_GPU_VECTOR_DOUBLE_POLICY_HPP
#define INCLUDE_VECTOR_DETAIL_GPU_VECTOR_DOUBLE_POLICY_HPP
#include "simulator/vector/detail/gpu_vector_policy.cuh"

namespace mindquantum::sim::vector::detail {
struct GPUVectorPolicyDouble : public GPUVectorPolicyBase<GPUVectorPolicyDouble, double> {};
}  // namespace mindquantum::sim::vector::detail
#endif
