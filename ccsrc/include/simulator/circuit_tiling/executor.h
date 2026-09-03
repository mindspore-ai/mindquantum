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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_EXECUTOR_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_EXECUTOR_H_

#include <complex>
#include <string_view>
#include <vector>

#include "core/mq_base_types.h"
#include "simulator/circuit_tiling/circuit_tiling.h"

// The circuit-tiling execution engine shared by every CPU simulator whose
// state is a flat array of complex amplitudes.  The best instruction set
// available on the host is selected once at runtime.
namespace mindquantum::sim::circuit_tiling {
// Applies a prepared tile in place.  `parallel` selects an OpenMP team over
// the tile traversal; small states run serially on the calling thread.
template <typename value_t>
void ApplyTile(value_t* state, const PreparedTile<value_t>& tile, bool parallel);

// Runs an adjoint program: evolves the shared bra and every batch ket while
// accumulating the initial overlaps and the derivative contributions into
// `output[batch][column]`.
template <typename value_t>
void ApplyAdjointProgram(value_t* shared_state, const std::vector<value_t*>& batch_states,
                         const AdjointProgram<value_t>& program, VVT<value_t>* output, bool conjugate_output,
                         bool parallel);

// The runtime selected for the value type: "scalar" or "arm-sve".
template <typename value_t>
[[nodiscard]] auto ExecutionBackendName() noexcept -> std::string_view;

extern template auto ExecutionBackendName<std::complex<float>>() noexcept -> std::string_view;
extern template auto ExecutionBackendName<std::complex<double>>() noexcept -> std::string_view;

extern template void ApplyTile<std::complex<float>>(std::complex<float>*, const PreparedTile<std::complex<float>>&,
                                                    bool);
extern template void ApplyTile<std::complex<double>>(std::complex<double>*, const PreparedTile<std::complex<double>>&,
                                                     bool);
extern template void ApplyAdjointProgram<std::complex<float>>(std::complex<float>*,
                                                              const std::vector<std::complex<float>*>&,
                                                              const AdjointProgram<std::complex<float>>&,
                                                              VVT<std::complex<float>>*, bool, bool);
extern template void ApplyAdjointProgram<std::complex<double>>(std::complex<double>*,
                                                               const std::vector<std::complex<double>*>&,
                                                               const AdjointProgram<std::complex<double>>&,
                                                               VVT<std::complex<double>>*, bool, bool);
}  // namespace mindquantum::sim::circuit_tiling

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_EXECUTOR_H_
