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

#ifndef INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP
#define INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/hamiltonian.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/circuit_tiling/state_space.h"

namespace mindquantum::sim::densitymatrix::detail {
// A mixed quantum state stored as a dense density matrix.
//
// Gates and noise channels are executed as circuit tiles on the row-major
// matrix viewed as a 2n-qubit state vector, so the same planning, lowering,
// and vectorised execution serve both this simulator and the state vector
// simulator.  Gradients use adjoint differentiation: the observable is
// evolved in the Heisenberg picture while the state is evolved backwards
// through unitary segments and re-simulated across noise channels.
template <typename qs_policy_t_>
class DensityMatrixState {
 public:
    using qs_policy_t = qs_policy_t_;
    using calc_type = typename qs_policy_t::calc_type;
    using derived_t = DensityMatrixState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using qs_data_p_t = typename qs_policy_t::qs_data_p_t;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using py_qs_datas_t = typename qs_policy_t::py_qs_datas_t;
    using matrix_t = typename qs_policy_t::matrix_t;
    using RndEngine = std::mt19937;

    explicit DensityMatrixState(qbit_t n_qubits, unsigned seed = 42);
    // Adopts ownership of a state allocated by the policy.
    DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed);

    DensityMatrixState(const DensityMatrixState& other);
    auto operator=(const DensityMatrixState& other) -> DensityMatrixState&;
    DensityMatrixState(DensityMatrixState&& other) noexcept = default;
    auto operator=(DensityMatrixState&& other) noexcept -> DensityMatrixState& = default;
    ~DensityMatrixState() = default;

    [[nodiscard]] auto DType() const -> tensor::TDtype;

    //! Reset the quantum state to the zero state |0...0><0...0|
    void Reset();

    void SetSeed(unsigned seed);

    //! Display basic information of this quantum state
    void Display(qbit_t qubits_limit = 10) const;

    [[nodiscard]] auto GetQS() const -> matrix_t;
    //! Set the state to the pure state |psi><psi|
    void SetQS(const py_qs_datas_t& state_vector);
    void SetDM(const matrix_t& density_matrix);

    [[nodiscard]] auto Purity() const -> calc_type;
    [[nodiscard]] auto GetPartialTrace(const qbits_t& traced_qubits) const -> matrix_t;
    [[nodiscard]] auto GetReducedDensityMatrix(const qbits_t& kept_qubits) const -> matrix_t;
    //! The state vector of a pure density matrix, up to a global phase
    [[nodiscard]] auto PureStateVector() const -> py_qs_datas_t;

    //! rho -> H rho H^dagger
    void ApplyHamiltonian(const Hamiltonian<calc_type>& ham);

    /*!
     * \brief Apply a quantum gate, a measurement, or a noise channel.  With
     * `diff` the derivative of a single-parameter gate is applied on both
     * sides: rho -> dU rho dU^dagger.  Returns the outcome of a measurement.
     */
    auto ApplyGate(const std::shared_ptr<BasicGate>& gate,
                   const parameter::ParameterResolver& pr = parameter::ParameterResolver(), bool diff = false)
        -> index_t;
    auto ApplyMeasure(const std::shared_ptr<BasicGate>& gate) -> index_t;
    auto ApplyCircuit(const circuit_t& circ, const parameter::ParameterResolver& pr = parameter::ParameterResolver())
        -> std::map<std::string, int>;

    [[nodiscard]] auto GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                      const parameter::ParameterResolver& pr) const -> py_qs_data_t;

    //! Expectation values and gradients of every Hamiltonian for every
    //! encoder parameter set, batched over `mea_threads` observables.
    [[nodiscard]] auto GetExpectationWithGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
        size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>>;

    [[nodiscard]] auto Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr, size_t shots,
                                const MST<size_t>& key_map, unsigned seed) const -> VT<unsigned>;

    //! Samples a circuit whose measurements all come last, from one exact
    //! evolution of the (possibly noisy) circuit.
    [[nodiscard]] auto SamplingMeasurementEnding(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                 size_t shots, const MST<size_t>& key_map, unsigned seed) const
        -> VT<unsigned>;

    template <typename policy_des, template <typename p_src, typename p_des> class cast_policy>
    [[nodiscard]] auto astype(unsigned new_seed) const -> DensityMatrixState<policy_des> {
        return DensityMatrixState<policy_des>(cast_policy<qs_policy_t, policy_des>::cast(qs_.get(), dim_), n_qubits_,
                                              new_seed);
    }

 private:
    struct StateDeleter {
        void operator()(qs_data_t* qs) const noexcept {
            qs_policy_t::FreeState(qs);
        }
    };
    using state_ptr_t = std::unique_ptr<qs_data_t, StateDeleter>;
    using prepared_tile_t = circuit_tiling::PreparedTile<py_qs_data_t>;
    using adjoint_program_t = circuit_tiling::AdjointProgram<py_qs_data_t>;

    // A maximal run of unitary gates followed by at most one noise channel.
    struct GradientSegment {
        circuit_t gates;
        std::shared_ptr<BasicGate> channel;
        bool has_gradient = false;
        circuit_tiling::Schedule forward;
        circuit_tiling::Schedule adjoint;
    };

    // The circuit structure of a gradient evaluation, shared by every
    // parameter set of a batch.
    struct GradientPlan {
        std::vector<GradientSegment> segments;
        size_t first_gradient_segment = 0;
    };

    // The lowering of one segment for one parameter set.
    struct PreparedSegment {
        std::vector<prepared_tile_t> forward;
        std::optional<prepared_tile_t> channel;
        std::optional<prepared_tile_t> adjoint_channel;
        // Gradient segments run an adjoint program; the others only evolve
        // the observables.
        adjoint_program_t program;
        std::vector<prepared_tile_t> observable_evolution;
    };

    // Observables evolved in one adjoint sweep are full density matrices;
    // batching them bounds the working set.
    static constexpr size_t kMaxObservableBatch = 16;

    // Validates the qubit count against the state index range.
    [[nodiscard]] static auto Dimension(qbit_t n_qubits) -> index_t;
    [[nodiscard]] auto Space() const -> circuit_tiling::StateSpace;
    [[nodiscard]] auto Random() const -> double;
    [[nodiscard]] auto CloneState() const -> state_ptr_t;
    [[nodiscard]] static auto CloneState(const qs_data_p_t& qs, index_t dim) -> state_ptr_t;
    [[nodiscard]] static auto Normalize(const circuit_t& circ) -> circuit_t;

    [[nodiscard]] auto SingleGateTile(const std::shared_ptr<BasicGate>& gate) const -> circuit_tiling::CircuitTile;
    [[nodiscard]] auto PlanCircuit(const circuit_t& gates, bool adjoint) const -> circuit_tiling::Schedule;
    [[nodiscard]] auto PrepareSchedule(const circuit_tiling::Schedule& schedule,
                                       const parameter::ParameterResolver& pr) const -> std::vector<prepared_tile_t>;
    static void ApplyTileTo(qs_data_p_t qs, const prepared_tile_t& tile, index_t dim);
    void ApplyTiledGate(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr, bool diff);

    [[nodiscard]] auto PlanGradient(const circuit_t& circ) const -> GradientPlan;
    [[nodiscard]] auto PrepareSegments(const GradientPlan& plan, const parameter::ParameterResolver& pr,
                                       const MST<size_t>& parameter_map) const -> std::vector<PreparedSegment>;
    void ApplyForward(qs_data_p_t qs, const std::vector<PreparedSegment>& segments, size_t through_segment) const;
    [[nodiscard]] auto ExpectationWithGradient(const GradientPlan& plan,
                                               const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
                                               const parameter::ParameterResolver& pr, const MST<size_t>& parameter_map,
                                               size_t batch_size) const -> VVT<py_qs_data_t>;
    void AdjointSweep(const GradientPlan& plan, const std::vector<PreparedSegment>& segments,
                      const qs_data_p_t& final_state, const std::vector<qs_data_p_t>& observables,
                      VVT<py_qs_data_t>* output) const;

    state_ptr_t qs_;
    qbit_t n_qubits_ = 0;
    index_t dim_ = 0;
    unsigned seed_ = 0;
    mutable RndEngine rnd_eng_;
};
}  // namespace mindquantum::sim::densitymatrix::detail

#include "simulator/densitymatrix/densitymatrix_state.tpp"  // NOLINT

#endif
