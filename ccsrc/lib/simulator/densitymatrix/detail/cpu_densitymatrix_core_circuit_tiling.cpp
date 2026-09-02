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

#include "config/openmp.h"
#include "core/utils.h"
#include "simulator/circuit_tiling/executor.h"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::ApplyCircuitTile(qs_data_p_t qs,
                                                          const circuit_tiling::PreparedTile<py_qs_data_t>& tile,
                                                          index_t dim) {
    circuit_tiling::ApplyTile(qs, tile, Parallel(dim));
}

template <typename calc_type_>
void CPUDensityMatrixPolicy<calc_type_>::ApplyAdjointTileProgram(
    qs_data_p_t shared_state, const std::vector<qs_data_p_t>& batch_states,
    const circuit_tiling::AdjointProgram<py_qs_data_t>& program, index_t dim, VVT<py_qs_data_t>* output) {
    circuit_tiling::ApplyAdjointProgram(shared_state, batch_states, program, output, false, Parallel(dim));
}

template struct CPUDensityMatrixPolicy<float>;
template struct CPUDensityMatrixPolicy<double>;
}  // namespace mindquantum::sim::densitymatrix::detail
