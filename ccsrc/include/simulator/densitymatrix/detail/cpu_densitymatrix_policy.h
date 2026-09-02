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

#ifndef INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP
#define INCLUDE_DENSITYMATRIX_DETAIL_CPU_DENSITYMATRIX_POLICY_HPP

#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/tensor/ops_cpu/utils.h"
#include "math/tensor/traits.h"
#include "ops/hamiltonian.h"
#include "simulator/circuit_tiling/circuit_tiling.h"

namespace mindquantum::sim::densitymatrix::detail {
// Storage and full-state kernels of the CPU density matrix simulator.
//
// A state is a dense row-major dim x dim matrix: rho[i][j] is stored at
// i * dim + j, so the state is also a state vector over 2n index bits whose
// low n bits are the column index and whose high n bits are the row index.
// Every gate and channel is executed by the shared circuit-tiling engine on
// that vector; only measurement, observables, and state access need the
// full-state kernels declared here.
template <typename calc_type_>
struct CPUDensityMatrixPolicy {
    using calc_type = calc_type_;
    using qs_data_t = std::complex<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = std::complex<calc_type>;
    using py_qs_datas_t = std::vector<py_qs_data_t>;
    using matrix_t = std::vector<py_qs_datas_t>;
    static constexpr tensor::TDtype dtype = tensor::to_dtype_v<py_qs_data_t>;
    // OpenMP threshold on the number of stored amplitudes.
    static constexpr index_t DimTh = static_cast<index_t>(1) << nQubitTh;

    [[nodiscard]] static constexpr auto Size(index_t dim) noexcept -> index_t {
        return dim * dim;
    }

    [[nodiscard]] static constexpr auto Parallel(index_t dim) noexcept -> bool {
        return Size(dim) >= DimTh;
    }

    // Storage
    // ========================================================================================================

    [[nodiscard]] static auto InitState(index_t dim, bool zero_state = true) -> qs_data_p_t;
    static void FreeState(qs_data_p_t qs) noexcept;
    [[nodiscard]] static auto Copy(const qs_data_p_t& qs, index_t dim) -> qs_data_p_t;
    static void CopyInto(qs_data_p_t des, const qs_data_p_t& src, index_t dim);
    static void SetZeroState(qs_data_p_t qs, index_t dim);

    // State access
    // ========================================================================================================

    static void Display(const qs_data_p_t& qs, qbit_t n_qubits, qbit_t q_limit);
    [[nodiscard]] static auto GetQS(const qs_data_p_t& qs, index_t dim) -> matrix_t;
    static void SetQS(qs_data_p_t qs, const py_qs_datas_t& state_vector, index_t dim);
    static void SetDM(qs_data_p_t qs, const matrix_t& density_matrix, index_t dim);
    [[nodiscard]] static auto Purity(const qs_data_p_t& qs, index_t dim) -> calc_type;
    // Traces out every qubit not listed in `kept_qubits` (strictly ascending).
    [[nodiscard]] static auto ReducedDensityMatrix(const qs_data_p_t& qs, const qbits_t& kept_qubits, index_t dim)
        -> matrix_t;
    [[nodiscard]] static auto PureStateVector(const qs_data_p_t& qs, index_t dim) -> py_qs_datas_t;
    // The Hilbert-Schmidt inner product Tr(bra^dagger ket).
    [[nodiscard]] static auto InnerProduct(const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim) -> py_qs_data_t;

    // Circuit tiles
    // ========================================================================================================

    static void ApplyCircuitTile(qs_data_p_t qs, const circuit_tiling::PreparedTile<py_qs_data_t>& tile, index_t dim);
    static void ApplyAdjointTileProgram(qs_data_p_t shared_state, const std::vector<qs_data_p_t>& batch_states,
                                        const circuit_tiling::AdjointProgram<py_qs_data_t>& program, index_t dim,
                                        VVT<py_qs_data_t>* output);

    // Measurement
    // ========================================================================================================

    [[nodiscard]] static auto OutcomeProbability(const qs_data_p_t& qs, qbit_t qubit, index_t dim) -> calc_type;
    static void Collapse(qs_data_p_t qs, qbit_t qubit, bool outcome, calc_type probability, index_t dim);
    [[nodiscard]] static auto DiagonalCumulativeProbabilities(const qs_data_p_t& qs, index_t dim) -> VT<calc_type>;

    // Observables
    // ========================================================================================================

    [[nodiscard]] static auto HamiltonianMatrix(const Hamiltonian<calc_type>& ham, index_t dim) -> qs_data_p_t;
    [[nodiscard]] static auto Expectation(const qs_data_p_t& qs, const Hamiltonian<calc_type>& ham, index_t dim)
        -> py_qs_data_t;
    // rho -> H rho H^dagger
    static void ApplyHamiltonian(qs_data_p_t qs, const Hamiltonian<calc_type>& ham, index_t dim);
};

using CPUDensityMatrixPolicyFloat = CPUDensityMatrixPolicy<float>;
using CPUDensityMatrixPolicyDouble = CPUDensityMatrixPolicy<double>;

template <typename policy_src, typename policy_des>
struct CastTo {
    static typename policy_des::qs_data_p_t cast(typename policy_src::qs_data_p_t qs, index_t dim) {
        if constexpr (std::is_same_v<policy_src, policy_des>) {
            return policy_des::Copy(qs, dim);
        } else {
            auto des = policy_des::InitState(dim, false);
            auto caster = tensor::cast_value<typename policy_src::calc_type, typename policy_des::calc_type>();
            const auto size = policy_des::Size(dim);
            THRESHOLD_OMP_FOR(
                size, policy_des::DimTh, for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(size); i++) {
                    des[i] = typename policy_des::qs_data_t{caster(std::real(qs[i])), caster(std::imag(qs[i]))};
                })
            return des;
        }
    }
};
}  // namespace mindquantum::sim::densitymatrix::detail

#endif
