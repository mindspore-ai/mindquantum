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

#include "simulator/circuit_tiling/executor.h"

#include <complex>
#include <string_view>
#include <vector>

#include "simulator/circuit_tiling/detail/scalar_runtime.h"
#include "simulator/circuit_tiling/detail/tile_engine.h"
#include "simulator/cpu_features.h"
#if MQ_HAS_ARM_SVE_BACKEND
#    include "arm_sve/sve_executor.h"
#endif

namespace mindquantum::sim::circuit_tiling {
namespace {
// The SVE runtime pays for gathers and scatters on every amplitude, which
// only pays off once a vector carries at least two complex amplitudes of the
// value type; measured: 1.4x faster than the scalar runtime with two lanes,
// 1.4x slower with one.
template <typename value_t>
[[nodiscard]] auto UseArmSve() noexcept -> bool {
#if MQ_HAS_ARM_SVE_BACKEND
    static const bool selected = cpu::HostHasArmSveFcma() && cpu::ArmSveVectorLengthBytes() >= 2 * sizeof(value_t);
    return selected;
#else
    return false;
#endif
}
}  // namespace

template <typename value_t>
void ApplyTile(value_t* state, const PreparedTile<value_t>& tile, int threads) {
#if MQ_HAS_ARM_SVE_BACKEND
    if (UseArmSve<value_t>()) {
        arm_sve::ApplyTile(state, tile, threads);
        return;
    }
#endif
    detail::ApplyTile<detail::ScalarTileRuntime<value_t>>(state, tile, threads);
}

template <typename value_t>
void ApplyAdjointProgram(value_t* shared_state, const std::vector<value_t*>& batch_states,
                         const AdjointProgram<value_t>& program, VVT<value_t>* output, bool conjugate_output,
                         int threads) {
#if MQ_HAS_ARM_SVE_BACKEND
    if (UseArmSve<value_t>()) {
        arm_sve::ApplyAdjointProgram(shared_state, batch_states, program, output, conjugate_output, threads);
        return;
    }
#endif
    detail::ApplyAdjointProgram<detail::ScalarTileRuntime<value_t>>(shared_state, batch_states, program, output,
                                                                    conjugate_output, threads);
}

template <typename value_t>
auto ExecutionBackendName() noexcept -> std::string_view {
    return UseArmSve<value_t>() ? "arm-sve" : "scalar";
}

template auto ExecutionBackendName<std::complex<float>>() noexcept -> std::string_view;
template auto ExecutionBackendName<std::complex<double>>() noexcept -> std::string_view;
template void ApplyTile<std::complex<float>>(std::complex<float>*, const PreparedTile<std::complex<float>>&, int);
template void ApplyTile<std::complex<double>>(std::complex<double>*, const PreparedTile<std::complex<double>>&, int);
template void ApplyAdjointProgram<std::complex<float>>(std::complex<float>*, const std::vector<std::complex<float>*>&,
                                                       const AdjointProgram<std::complex<float>>&,
                                                       VVT<std::complex<float>>*, bool, int);
template void ApplyAdjointProgram<std::complex<double>>(std::complex<double>*,
                                                        const std::vector<std::complex<double>*>&,
                                                        const AdjointProgram<std::complex<double>>&,
                                                        VVT<std::complex<double>>*, bool, int);
}  // namespace mindquantum::sim::circuit_tiling
