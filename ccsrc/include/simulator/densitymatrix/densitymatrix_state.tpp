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

#ifndef INCLUDE_DENSITYMATRIX_DENSITYMATRIXSTATE_TPP
#define INCLUDE_DENSITYMATRIX_DENSITYMATRIXSTATE_TPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/circuit_tiling/builder.h"
#include "simulator/circuit_tiling/planner.h"
#include "simulator/circuit_tiling/state_space.h"
#include "simulator/densitymatrix/densitymatrix_state.h"
#include "simulator/utils.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Dimension(qbit_t n_qubits) -> index_t {
    return index_t{1} << circuit_tiling::StateSpace::DensityMatrix(n_qubits).circuit_qubits();
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qbit_t n_qubits, unsigned seed)
    : qs_(qs_policy_t::InitState(Dimension(n_qubits)))
    , n_qubits_(n_qubits)
    , dim_(Dimension(n_qubits))
    , seed_(seed)
    , rnd_eng_(seed) {
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed)
    : qs_(qs), n_qubits_(n_qubits), dim_(Dimension(n_qubits)), seed_(seed), rnd_eng_(seed) {
    if (qs == nullptr) {
        throw std::invalid_argument("Cannot adopt a null density matrix.");
    }
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(const DensityMatrixState& other)
    : qs_(qs_policy_t::Copy(other.qs_.get(), other.dim_))
    , n_qubits_(other.n_qubits_)
    , dim_(other.dim_)
    , seed_(other.seed_)
    , rnd_eng_(other.seed_) {
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::operator=(const DensityMatrixState& other) -> DensityMatrixState& {
    if (this != &other) {
        qs_.reset(qs_policy_t::Copy(other.qs_.get(), other.dim_));
        n_qubits_ = other.n_qubits_;
        dim_ = other.dim_;
        SetSeed(other.seed_);
    }
    return *this;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Space() const -> circuit_tiling::StateSpace {
    return circuit_tiling::StateSpace::DensityMatrix(n_qubits_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Random() const -> double {
    return std::uniform_real_distribution<double>(0., 1.)(rnd_eng_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::CloneState() const -> state_ptr_t {
    return CloneState(qs_.get(), dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::CloneState(const qs_data_p_t& qs, index_t dim) -> state_ptr_t {
    return state_ptr_t(qs_policy_t::Copy(qs, dim));
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Normalize(const circuit_t& circ) -> circuit_t {
    circuit_t expanded;
    expanded.reserve(circ.size());
    for (const auto& gate : circ) {
        if (gate == nullptr) {
            throw std::invalid_argument("Cannot apply a null gate.");
        }
        if (gate->GetID() != GateID::GPL) {
            expanded.push_back(gate);
            continue;
        }
        // A grouped Pauli channel is a product of single-qubit Pauli
        // channels; the planner tiles those independently.
        const auto* grouped = static_cast<const GroupedPauliChannel*>(gate.get());
        for (auto channel = grouped->begin(); channel != grouped->end(); ++channel) {
            expanded.push_back(std::make_shared<PauliChannel>(*channel));
        }
    }
    return expanded;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::DType() const -> tensor::TDtype {
    return qs_policy_t::dtype;
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::Reset() {
    qs_policy_t::SetZeroState(qs_.get(), dim_);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::SetSeed(unsigned seed) {
    seed_ = seed;
    rnd_eng_.seed(seed);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::Display(qbit_t qubits_limit) const {
    qs_policy_t::Display(qs_.get(), n_qubits_, qubits_limit);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetQS() const -> matrix_t {
    return qs_policy_t::GetQS(qs_.get(), dim_);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::SetQS(const py_qs_datas_t& state_vector) {
    qs_policy_t::SetQS(qs_.get(), state_vector, dim_);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::SetDM(const matrix_t& density_matrix) {
    qs_policy_t::SetDM(qs_.get(), density_matrix, dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Purity() const -> calc_type {
    return qs_policy_t::Purity(qs_.get(), dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetPartialTrace(const qbits_t& traced_qubits) const -> matrix_t {
    std::vector<bool> traced(static_cast<size_t>(n_qubits_), false);
    for (auto qubit : traced_qubits) {
        if (qubit < 0 || qubit >= n_qubits_) {
            throw std::invalid_argument(fmt::format("Qubit {} is out of range.", qubit));
        }
        traced[static_cast<size_t>(qubit)] = true;
    }
    qbits_t kept;
    for (qbit_t qubit = 0; qubit < n_qubits_; ++qubit) {
        if (!traced[static_cast<size_t>(qubit)]) {
            kept.push_back(qubit);
        }
    }
    return qs_policy_t::ReducedDensityMatrix(qs_.get(), kept, dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetReducedDensityMatrix(const qbits_t& kept_qubits) const -> matrix_t {
    auto kept = kept_qubits;
    std::sort(kept.begin(), kept.end());
    return qs_policy_t::ReducedDensityMatrix(qs_.get(), kept, dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PureStateVector() const -> py_qs_datas_t {
    return qs_policy_t::PureStateVector(qs_.get(), dim_);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    qs_policy_t::ApplyHamiltonian(qs_.get(), ham, dim_);
}

// Gate application
// ============================================================================

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GateTile(const std::shared_ptr<BasicGate>& gate, bool derivative) const
    -> circuit_tiling::CircuitTile {
    const auto space = Space();
    const auto access = circuit_tiling::DescribeGateAccess(gate, space);
    if (!access.has_value()) {
        throw std::invalid_argument(fmt::format("Apply of gate {} not implement.", gate->GetID()));
    }
    const auto count = [](index_t mask) { return CountOne(static_cast<std::uint64_t>(mask)); };
    const auto footprint = access->mixing_mask | access->object_mask | access->control_mask;
    // A derivative is zero outside its control subspace, which the tile can
    // only express when every control is local: its tile is the whole
    // footprint however wide.  A gate keeps diagonal operands and controls
    // local when they fit, so that the hot loop resolves them without
    // external masks.
    auto mask = derivative || count(footprint) <= circuit_tiling::kMaxTileQubits ? footprint : access->mixing_mask;
    // Further image pairs fill the tile so that the per-block overhead is
    // amortised over full-width, image-symmetric blocks.
    for (qbit_t qubit = 0; qubit < n_qubits_ && count(mask) + 2 <= circuit_tiling::kMaxTileQubits; ++qubit) {
        mask |= space.Embed(index_t{1} << qubit);
    }
    return circuit_tiling::CircuitTile{circuit_tiling::QubitsFromMask(mask), {gate}};
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PlanCircuit(const circuit_t& gates, bool adjoint) const
    -> circuit_tiling::Schedule {
    const circuit_tiling::Planner planner(Space());
    return adjoint ? planner.PlanAdjoint(gates) : planner.Plan(gates);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PrepareSchedule(const circuit_tiling::Schedule& schedule,
                                                       const parameter::ParameterResolver& pr) const
    -> std::vector<prepared_tile_t> {
    using builder_t = circuit_tiling::Builder<py_qs_data_t>;
    const auto space = Space();
    std::vector<prepared_tile_t> prepared;
    prepared.reserve(schedule.size());
    for (const auto& entry : schedule) {
        if (const auto* tile = std::get_if<circuit_tiling::CircuitTile>(&entry); tile != nullptr) {
            prepared.push_back(builder_t::Build(*tile, pr, space));
        } else {
            prepared.push_back(
                builder_t::Build(GateTile(std::get<std::shared_ptr<BasicGate>>(entry), false), pr, space));
        }
    }
    return prepared;
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyTileTo(qs_data_p_t qs, const prepared_tile_t& tile, index_t dim) {
    if (!tile.operations.empty()) {
        qs_policy_t::ApplyCircuitTile(qs, tile, dim);
    }
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyTiledGate(const std::shared_ptr<BasicGate>& gate,
                                                      const parameter::ParameterResolver& pr, bool diff) {
    using builder_t = circuit_tiling::Builder<py_qs_data_t>;
    if (diff && gate->GradRequired()) {
        ApplyTileTo(qs_.get(), builder_t::BuildDerivative(GateTile(gate, true), pr, Space()), dim_);
    } else {
        ApplyTileTo(qs_.get(), builder_t::Build(GateTile(gate, false), pr, Space()), dim_);
    }
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate>& gate,
                                                 const parameter::ParameterResolver& pr, bool diff) -> index_t {
    if (gate == nullptr) {
        throw std::invalid_argument("Cannot apply a null gate.");
    }
    switch (gate->GetID()) {
        case GateID::I:
            break;
        case GateID::M:
            return ApplyMeasure(gate);
        case GateID::GPL:
            for (const auto& channel : Normalize({gate})) {
                ApplyTiledGate(channel, pr, false);
            }
            break;
        default:
            ApplyTiledGate(gate, pr, diff);
            break;
    }
    return 2;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyMeasure(const std::shared_ptr<BasicGate>& gate) -> index_t {
    const auto* measure = static_cast<const MeasureGate*>(gate.get());
    const auto objects = gate->GetObjQubits();
    if (objects.size() != 1 || objects.front() < 0 || objects.front() >= n_qubits_) {
        throw std::invalid_argument("A measurement acts on exactly one qubit of the simulator.");
    }
    const auto qubit = objects.front();
    const auto one = qs_policy_t::OutcomeProbability(qs_.get(), qubit, dim_);
    const bool outcome = Random() < one;
    qs_policy_t::Collapse(qs_.get(), qubit, outcome, outcome ? one : 1 - one, dim_);
    if (measure->WillReset() && static_cast<index_t>(outcome) != measure->GetRestTo()) {
        ApplyTiledGate(std::make_shared<XGate>(qbits_t{qubit}), parameter::ParameterResolver(), false);
    }
    return static_cast<index_t>(outcome);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyCircuit(const circuit_t& circ, const parameter::ParameterResolver& pr)
    -> std::map<std::string, int> {
    std::map<std::string, int> result;
    const auto circuit = Normalize(circ);
    const auto apply_gate = [this, &pr, &result](const std::shared_ptr<BasicGate>& gate) {
        if (gate->GetID() == GateID::M) {
            result[static_cast<MeasureGate*>(gate.get())->Name()] = ApplyMeasure(gate);
        } else {
            ApplyGate(gate, pr, false);
        }
    };
    if (circuit.size() < 2) {
        for (const auto& gate : circuit) {
            apply_gate(gate);
        }
        return result;
    }
    // Planning is cheap relative to lowering and executing tiles, and every
    // merged gate saves a state pass, so circuits are always planned.
    const auto space = Space();
    for (const auto& entry : PlanCircuit(circuit, false)) {
        if (const auto* tile = std::get_if<circuit_tiling::CircuitTile>(&entry); tile != nullptr) {
            ApplyTileTo(qs_.get(), circuit_tiling::Builder<py_qs_data_t>::Build(*tile, pr, space), dim_);
        } else {
            apply_gate(std::get<std::shared_ptr<BasicGate>>(entry));
        }
    }
    return result;
}

// Observables
// ============================================================================

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                      const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    derived_t evolved(*this);
    evolved.SetSeed(static_cast<unsigned>(Random() * (1 << 20)));
    evolved.ApplyCircuit(circ, pr);
    return qs_policy_t::Expectation(evolved.qs_.get(), ham, dim_);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PlanGradient(const circuit_t& circ) const -> GradientPlan {
    GradientPlan plan;
    plan.segments.emplace_back();
    for (const auto& gate : Normalize(circ)) {
        if (gate->GetID() == GateID::M) {
            throw std::invalid_argument("A circuit for gradient evaluation cannot contain measurements.");
        }
        if (circuit_tiling::IsNoiseChannel(gate->GetID())) {
            plan.segments.back().channel = gate;
            plan.segments.emplace_back();
            continue;
        }
        auto& segment = plan.segments.back();
        segment.gates.push_back(gate);
        segment.has_gradient = segment.has_gradient || gate->GradRequired();
    }

    const auto first_gradient = std::find_if(plan.segments.begin(), plan.segments.end(),
                                             [](const auto& segment) { return segment.has_gradient; });
    plan.first_gradient_segment = static_cast<size_t>(std::distance(plan.segments.begin(), first_gradient));

    for (size_t index = 0; index < plan.segments.size(); ++index) {
        auto& segment = plan.segments[index];
        segment.forward = PlanCircuit(segment.gates, false);
        if (index < plan.first_gradient_segment) {
            continue;
        }
        circuit_t reversed(segment.gates.rbegin(), segment.gates.rend());
        if (index == plan.first_gradient_segment) {
            // Nothing before the first differentiated gate contributes.
            const auto last_gradient = std::find_if(reversed.rbegin(), reversed.rend(),
                                                    [](const auto& gate) { return gate->GradRequired(); });
            reversed.erase(last_gradient.base(), reversed.end());
        }
        segment.adjoint = PlanCircuit(reversed, true);
    }
    return plan;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PrepareSegments(const GradientPlan& plan, const parameter::ParameterResolver& pr,
                                                       const MST<size_t>& parameter_map) const
    -> std::vector<PreparedSegment> {
    using builder_t = circuit_tiling::Builder<py_qs_data_t>;
    constexpr auto convention = builder_t::AdjointGates::Reversed;
    const auto space = Space();
    const auto build_adjoint = [&](const circuit_tiling::ScheduleEntry& entry) {
        if (const auto* tile = std::get_if<circuit_tiling::CircuitTile>(&entry); tile != nullptr) {
            return builder_t::BuildAdjoint(*tile, pr, parameter_map, space, convention);
        }
        return builder_t::BuildAdjoint(GateTile(std::get<std::shared_ptr<BasicGate>>(entry), false), pr, parameter_map,
                                       space, convention);
    };

    std::vector<PreparedSegment> prepared(plan.segments.size());
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        const auto& segment = plan.segments[index];
        auto& lowered = prepared[index];
        lowered.forward = PrepareSchedule(segment.forward, pr);
        if (segment.channel != nullptr) {
            const auto tile = GateTile(segment.channel, false);
            lowered.channel = builder_t::Build(tile, pr, space);
            lowered.adjoint_channel = builder_t::EvolutionOnly(
                builder_t::BuildAdjoint(tile, pr, parameter_map, space, convention));
        }
        if (index < plan.first_gradient_segment) {
            continue;
        }
        if (segment.has_gradient) {
            lowered.program.output_size = 1 + parameter_map.size();
            for (const auto& entry : segment.adjoint) {
                lowered.program.tiles.push_back(build_adjoint(entry));
            }
        } else {
            for (const auto& entry : segment.adjoint) {
                lowered.observable_evolution.push_back(builder_t::EvolutionOnly(build_adjoint(entry)));
            }
        }
    }
    return prepared;
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyForward(qs_data_p_t qs, const std::vector<PreparedSegment>& segments,
                                                    size_t through_segment) const {
    for (size_t index = 0; index <= through_segment; ++index) {
        for (const auto& tile : segments[index].forward) {
            ApplyTileTo(qs, tile, dim_);
        }
        if (index < through_segment) {
            ApplyTileTo(qs, *segments[index].channel, dim_);
        }
    }
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::AdjointSweep(const GradientPlan& plan,
                                                    const std::vector<PreparedSegment>& segments,
                                                    const qs_data_p_t& final_state,
                                                    const std::vector<qs_data_p_t>& observables,
                                                    VVT<py_qs_data_t>* output) const {
    const auto last = plan.segments.size() - 1;
    auto shared = CloneState(final_state, dim_);
    // Whether `shared` holds the state at the end of the segment being entered.
    bool shared_current = true;
    for (size_t index = last + 1; index-- > plan.first_gradient_segment;) {
        const auto& segment = segments[index];
        if (index != last) {
            // Heisenberg evolution of the observables through the channel
            // that follows this segment.  The state cannot follow: channels
            // are not invertible.
            for (auto* observable : observables) {
                ApplyTileTo(observable, *segment.adjoint_channel, dim_);
            }
            shared_current = false;
        }
        if (!plan.segments[index].has_gradient) {
            for (const auto& tile : segment.observable_evolution) {
                for (auto* observable : observables) {
                    ApplyTileTo(observable, tile, dim_);
                }
            }
            shared_current = false;
            continue;
        }
        if (!shared_current) {
            shared = CloneState();
            ApplyForward(shared.get(), segments, index);
            shared_current = true;
        }
        qs_policy_t::ApplyAdjointTileProgram(shared.get(), observables, segment.program, dim_, output);
    }
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectationWithGradient(
    const GradientPlan& plan, const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
    const parameter::ParameterResolver& pr, const MST<size_t>& parameter_map, size_t batch_size) const
    -> VVT<py_qs_data_t> {
    const auto output_size = 1 + parameter_map.size();
    const auto segments = PrepareSegments(plan, pr, parameter_map);
    auto final_state = CloneState();
    ApplyForward(final_state.get(), segments, plan.segments.size() - 1);

    VVT<py_qs_data_t> result(hams.size(), VT<py_qs_data_t>(output_size, 0));
    for (size_t first = 0; first < hams.size(); first += batch_size) {
        const auto count = std::min(batch_size, hams.size() - first);
        std::vector<state_ptr_t> storage;
        std::vector<qs_data_p_t> observables;
        storage.reserve(count);
        observables.reserve(count);
        for (size_t offset = 0; offset < count; ++offset) {
            storage.emplace_back(qs_policy_t::HamiltonianMatrix(*hams[first + offset], dim_));
            observables.push_back(storage.back().get());
        }

        VVT<py_qs_data_t> group(count, VT<py_qs_data_t>(output_size, 0));
        for (size_t offset = 0; offset < count; ++offset) {
            group[offset][0] = qs_policy_t::InnerProduct(final_state.get(), observables[offset], dim_);
        }
        if (plan.first_gradient_segment < plan.segments.size()) {
            AdjointSweep(plan, segments, final_state.get(), observables, &group);
        }
        for (size_t offset = 0; offset < count; ++offset) {
            auto& row = result[first + offset];
            row[0] = group[offset][0];
            // The sweep accumulates Tr(rho dU^dagger H U); its conjugate is
            // the other half of the product rule.
            for (size_t column = 1; column < output_size; ++column) {
                row[column] = group[offset][column] + std::conj(group[offset][column]);
            }
        }
    }
    return result;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
    if (std::find(hams.begin(), hams.end(), nullptr) != hams.end()) {
        throw std::invalid_argument("Cannot evaluate a null Hamiltonian.");
    }
    if (mea_threads == 0) {
        throw std::invalid_argument("Observable batch size cannot be zero.");
    }
    MST<size_t> parameter_map;
    for (size_t index = 0; index < enc_name.size(); ++index) {
        parameter_map[enc_name[index]] = index;
    }
    for (size_t index = 0; index < ans_name.size(); ++index) {
        parameter_map[ans_name[index]] = index + enc_name.size();
    }
    const auto plan = PlanGradient(circ);
    // Every observable in a batch is evolved as a full density matrix, so
    // the batch bounds the working set to a fixed number of states.
    const auto batch_size = std::min(mea_threads, kMaxObservableBatch);

    VT<VVT<py_qs_data_t>> output(enc_data.size());
    const auto threads = SelectGradientBatchThreads(qs_policy_t::Size(dim_), qs_policy_t::DimTh, batch_threads);
    ParallelBatch(threads, enc_data.size(), [&](size_t batch) {
        parameter::ParameterResolver pr;
        pr.SetItems(enc_name, enc_data[batch]);
        pr.SetItems(ans_name, ans_data);
        output[batch] = ExpectationWithGradient(plan, hams, pr, parameter_map, batch_size);
    });
    return output;
}

// Sampling
// ============================================================================

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                size_t shots, const MST<size_t>& key_map, unsigned seed) const
    -> VT<unsigned> {
    const auto key_size = key_map.size();
    VT<unsigned> result(shots * key_size);
    RndEngine engine(seed);
    std::uniform_real_distribution<double> sub_seed(1.0, static_cast<double>(1 << 20));
    for (size_t shot = 0; shot < shots; ++shot) {
        derived_t trajectory(*this);
        trajectory.SetSeed(static_cast<unsigned>(sub_seed(engine)));
        for (const auto& [name, outcome] : trajectory.ApplyCircuit(circ, pr)) {
            result[shot * key_size + key_map.at(name)] = static_cast<unsigned>(outcome);
        }
    }
    return result;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::SamplingMeasurementEnding(const circuit_t& circ,
                                                                 const parameter::ParameterResolver& pr, size_t shots,
                                                                 const MST<size_t>& key_map, unsigned seed) const
    -> VT<unsigned> {
    std::vector<bool> measured(static_cast<size_t>(n_qubits_), false);
    std::vector<std::pair<qbit_t, size_t>> outputs;
    circuit_t evolution;
    for (const auto& gate : circ) {
        if (gate == nullptr) {
            throw std::invalid_argument("Cannot apply a null gate.");
        }
        if (gate->GetID() != GateID::M) {
            evolution.push_back(gate);
            continue;
        }
        const auto qubit = gate->GetObjQubits().at(0);
        if (qubit < 0 || qubit >= n_qubits_ || measured[static_cast<size_t>(qubit)]) {
            throw std::invalid_argument("Quantum circuit is not a measurement ending circuit.");
        }
        measured[static_cast<size_t>(qubit)] = true;
        outputs.emplace_back(qubit, key_map.at(static_cast<MeasureGate*>(gate.get())->Name()));
    }

    derived_t evolved(*this);
    evolved.ApplyCircuit(evolution, pr);
    const auto cumulative = qs_policy_t::DiagonalCumulativeProbabilities(evolved.qs_.get(), dim_);

    const auto key_size = key_map.size();
    VT<unsigned> result(shots * key_size);
    RndEngine engine(seed);
    std::uniform_real_distribution<double> uniform(0., 1.);
    for (size_t shot = 0; shot < shots; ++shot) {
        const auto sample = static_cast<calc_type>(uniform(engine));
        const auto basis = static_cast<index_t>(
            std::distance(cumulative.begin(), std::upper_bound(cumulative.begin(), cumulative.end(), sample)));
        const auto outcome = std::min(basis, dim_ - 1);
        for (const auto& [qubit, column] : outputs) {
            result[shot * key_size + column] = static_cast<unsigned>((outcome >> qubit) & 1);
        }
    }
    return result;
}
}  // namespace mindquantum::sim::densitymatrix::detail

#endif
