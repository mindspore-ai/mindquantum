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

#ifndef LIB_SIMULATOR_CIRCUIT_TILING_ARM_SVE_SVE_EXECUTOR_H_
#define LIB_SIMULATOR_CIRCUIT_TILING_ARM_SVE_SVE_EXECUTOR_H_

#include <complex>
#include <vector>

#include "core/mq_base_types.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/worker_scratch.h"

// Entry points of the translation unit compiled with Arm SVE/FCMA enabled.
// Only the runtime dispatcher may call them, after confirming host support.
namespace mindquantum::sim::circuit_tiling::arm_sve {
template <typename calc_type>
void ReserveScratch(WorkerScratch<calc_type>& scratch, size_t logical_size, int threads);

template <typename value_t>
void ApplyTile(value_t* state, const PreparedTile<value_t>& tile, int threads,
               WorkerScratch<typename value_t::value_type>& scratch);

template <typename value_t>
void ApplyAdjointProgram(value_t* shared_state, const std::vector<value_t*>& batch_states,
                         const AdjointProgram<value_t>& program, VVT<value_t>* output, bool conjugate_output,
                         int threads, WorkerScratch<typename value_t::value_type>& scratch);

extern template void ApplyTile<std::complex<float>>(std::complex<float>*, const PreparedTile<std::complex<float>>&, int,
                                                    WorkerScratch<float>&);
extern template void ApplyTile<std::complex<double>>(std::complex<double>*, const PreparedTile<std::complex<double>>&,
                                                     int, WorkerScratch<double>&);
extern template void ApplyAdjointProgram<std::complex<float>>(std::complex<float>*,
                                                              const std::vector<std::complex<float>*>&,
                                                              const AdjointProgram<std::complex<float>>&,
                                                              VVT<std::complex<float>>*, bool, int,
                                                              WorkerScratch<float>&);
extern template void ApplyAdjointProgram<std::complex<double>>(std::complex<double>*,
                                                               const std::vector<std::complex<double>*>&,
                                                               const AdjointProgram<std::complex<double>>&,
                                                               VVT<std::complex<double>>*, bool, int,
                                                               WorkerScratch<double>&);
}  // namespace mindquantum::sim::circuit_tiling::arm_sve

#endif  // LIB_SIMULATOR_CIRCUIT_TILING_ARM_SVE_SVE_EXECUTOR_H_
