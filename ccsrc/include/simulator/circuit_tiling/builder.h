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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_BUILDER_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_BUILDER_H_

#include <cmath>

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/circuit_tiling/state_space.h"

namespace mindquantum::sim::circuit_tiling {
// Lowers the gates of a circuit tile into the mask-free kernels executed by
// the tile engine.  A gate is lowered once per image of its qubits in the
// state space; a noise channel is lowered as a superoperator on both images
// of a density matrix.
template <typename value_t>
class Builder {
 public:
    using calc_type = typename value_t::value_type;

    // How the gates listed in an adjoint tile relate to the differentiated
    // circuit.
    enum class AdjointGates : std::uint8_t {
        // The tile lists the daggered gates of the reversed circuit, so each
        // lowering is applied as-is.
        Daggered,
        // The tile lists the original gates in reverse order, so the adjoint
        // of each lowering is applied.  Noise channels are adjoint-evolved as
        // well, which is the Heisenberg-picture evolution of an observable.
        Reversed,
    };

    [[nodiscard]] static auto Build(const CircuitTile& tile, const parameter::ParameterResolver& parameter_resolver,
                                    const StateSpace& space) -> PreparedTile<value_t> {
        ValidateTile(tile, space);
        PreparedTile<value_t> prepared;
        prepared.layout = BuildLayout(tile, space);
        prepared.operations.reserve(tile.gates.size() * space.Images().size());
        for (const auto& gate : tile.gates) {
            auto operations = LowerGate(*gate, tile.qubits, parameter_resolver, space);
            prepared.operations.insert(prepared.operations.end(), std::make_move_iterator(operations.begin()),
                                       std::make_move_iterator(operations.end()));
        }
        prepared.operations = FoldDiagonalRuns(std::move(prepared.operations), prepared.layout.dimension);
        for (const auto& operation : prepared.operations) {
            prepared.layout.workspace_size = std::max(prepared.layout.workspace_size, WorkspaceSize(operation.kernel));
        }
        return prepared;
    }

    [[nodiscard]] static auto BuildAdjoint(const CircuitTile& tile,
                                           const parameter::ParameterResolver& parameter_resolver,
                                           const MST<size_t>& parameter_map, const StateSpace& space,
                                           AdjointGates convention) -> PreparedAdjointTile<value_t> {
        ValidateTile(tile, space);
        PreparedAdjointTile<value_t> prepared;
        prepared.layout = BuildLayout(tile, space);
        const auto tile_dim = prepared.layout.dimension;
        prepared.operations.reserve(tile.gates.size());

        for (const auto& gate : tile.gates) {
            AdjointOperation<value_t> adjoint;
            if (IsNoiseChannel(gate->GetID())) {
                adjoint.evolution.push_back(PrepareOperation(BuildChannel(*gate, tile.qubits, space), tile_dim));
            } else {
                const auto images = space.Images();
                for (size_t image_index = 0; image_index < images.size(); ++image_index) {
                    const auto& image = images[image_index];
                    auto evolution = BuildOperation(*gate, tile.qubits, parameter_resolver, image);
                    if (image_index + 1 == images.size() && gate->GradRequired()) {
                        auto derivatives = BuildIntrinsicDerivatives(*gate, tile.qubits, parameter_resolver, evolution,
                                                                     image);
                        auto targets = BuildGradientTargets(*gate, parameter_map, derivatives.size());
                        for (size_t derivative_index = 0; derivative_index < derivatives.size(); ++derivative_index) {
                            if (targets[derivative_index].empty()) {
                                continue;
                            }
                            auto derivative = PrepareDerivative(std::move(derivatives[derivative_index]), tile_dim);
                            derivative.targets = std::move(targets[derivative_index]);
                            adjoint.derivatives.push_back(std::move(derivative));
                        }
                        adjoint.derivative_position = adjoint.evolution.size();
                    }
                    auto operation = PrepareOperation(std::move(evolution), tile_dim);
                    if (image.conjugated) {
                        Conjugate(&operation);
                    }
                    adjoint.evolution.push_back(std::move(operation));
                }
            }
            if (convention == AdjointGates::Reversed) {
                for (auto& operation : adjoint.evolution) {
                    Adjoint(&operation);
                }
                for (auto& derivative : adjoint.derivatives) {
                    Adjoint(&derivative);
                }
            }
            for (const auto& operation : adjoint.evolution) {
                prepared.layout.workspace_size = std::max(prepared.layout.workspace_size,
                                                          WorkspaceSize(operation.kernel));
            }
            for (const auto& derivative : adjoint.derivatives) {
                prepared.layout.workspace_size = std::max(prepared.layout.workspace_size,
                                                          WorkspaceSize(derivative.kernel));
            }
            prepared.operations.push_back(std::move(adjoint));
        }
        return prepared;
    }

    // Lowers the derivative of the single parameterized gate in `tile` on
    // every image as a linear map: on a density matrix this realises
    // dU rho dU^dagger.  A lowered kernel leaves the bases it does not mention
    // unchanged, which is right for a gate but not for its derivative: dU is
    // zero there (outside a control subspace, on the |0> of a phase shift, on
    // the |00> and |11> of an exchange).  The tile therefore ends with the
    // projection onto the support of the derivative. Every control must be a
    // tile qubit unless the caller projects it separately using projected_controls.
    [[nodiscard]] static auto BuildDerivative(const CircuitTile& tile,
                                              const parameter::ParameterResolver& parameter_resolver,
                                              const StateSpace& space,
                                              index_t projected_controls = 0) -> PreparedTile<value_t> {
        ValidateTile(tile, space);
        if (tile.gates.size() != 1) {
            throw std::invalid_argument("A derivative tile must hold exactly one gate.");
        }
        const auto& gate = *tile.gates.front();
        if (IsNoiseChannel(gate.GetID())) {
            throw std::invalid_argument(fmt::format("Gate {} has no derivative.", gate.GetID()));
        }
        PreparedTile<value_t> prepared;
        prepared.layout = BuildLayout(tile, space);
        const auto tile_dim = prepared.layout.dimension;
        std::vector<bool> support(tile_dim, true);
        for (const auto& image : space.Images()) {
            auto evolution = BuildOperation(gate, tile.qubits, parameter_resolver, image);
            auto derivatives = BuildIntrinsicDerivatives(gate, tile.qubits, parameter_resolver, evolution, image);
            if (derivatives.size() != 1) {
                throw std::invalid_argument(
                    fmt::format("Gate {} has {} intrinsic derivatives; only single-parameter derivatives can "
                                "be applied.",
                                gate.GetID(), derivatives.size()));
            }
            auto operation = PrepareOperation(std::move(derivatives.front()), tile_dim);
            operation.external_control_mask &= ~projected_controls;
            if (operation.external_control_mask != 0) {
                throw std::invalid_argument("A derivative tile must contain every control qubit of its gate.");
            }
            if (image.conjugated) {
                Conjugate(&operation);
            }
            const auto touched = TouchedBases(operation.kernel, tile_dim);
            for (size_t basis = 0; basis < tile_dim; ++basis) {
                support[basis] = support[basis] && touched[basis];
            }
            prepared.operations.push_back(std::move(operation));
        }
        DiagonalKernel<value_t> projection;
        for (size_t basis = 0; basis < tile_dim; ++basis) {
            if (!support[basis]) {
                projection.scales.push_back({basis, value_t{0, 0}});
            }
        }
        if (!projection.scales.empty()) {
            prepared.operations.push_back({0, std::move(projection)});
        }
        prepared.operations = FoldDiagonalRuns(std::move(prepared.operations), tile_dim);
        for (const auto& operation : prepared.operations) {
            prepared.layout.workspace_size = std::max(prepared.layout.workspace_size, WorkspaceSize(operation.kernel));
        }
        return prepared;
    }

    // The evolution of an adjoint tile without its derivatives.
    [[nodiscard]] static auto EvolutionOnly(PreparedAdjointTile<value_t> tile) -> PreparedTile<value_t> {
        PreparedTile<value_t> prepared;
        prepared.layout = std::move(tile.layout);
        for (auto& operation : tile.operations) {
            prepared.operations.insert(prepared.operations.end(), std::make_move_iterator(operation.evolution.begin()),
                                       std::make_move_iterator(operation.evolution.end()));
        }
        return prepared;
    }

    // Compose intrinsic derivative images without duplicating gate formulas.
    // Unlike gates, derivatives are zero outside their support. Jacobian
    // targets are applied by the caller, not by this linear map.
    [[nodiscard]] static auto DerivativeOnly(const TileLayout& layout,
                                             const Derivative<value_t>& derivative) -> PreparedTile<value_t> {
        if (derivative.external_control_mask != 0) {
            throw std::invalid_argument("A derivative image must contain every control qubit.");
        }
        PreparedTile<value_t> prepared;
        prepared.layout = layout;
        Operation<value_t> operation;
        operation.kernel = std::visit([](const auto& kernel) -> Kernel<value_t> { return kernel; }, derivative.kernel);
        prepared.layout.workspace_size = WorkspaceSize(operation.kernel);
        const auto touched = TouchedBases(operation.kernel, layout.dimension);
        prepared.operations.push_back(std::move(operation));
        DiagonalKernel<value_t> projection;
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            if (!touched[basis]) {
                projection.scales.push_back({basis, value_t{0, 0}});
            }
        }
        if (!projection.scales.empty()) {
            prepared.operations.push_back({0, std::move(projection)});
        }
        return prepared;
    }

    // Replaces a lowered linear map by its complex conjugate.
    static void Conjugate(Operation<value_t>* operation) {
        std::visit([](auto& kernel) { ConjugateKernel(&kernel); }, operation->kernel);
    }

    // Replaces a lowered linear map by its Hermitian adjoint.
    static void Adjoint(Operation<value_t>* operation) {
        std::visit([](auto& kernel) { AdjointKernel(&kernel); }, operation->kernel);
    }

    static void Adjoint(Derivative<value_t>* derivative) {
        std::visit([](auto& kernel) { AdjointKernel(&kernel); }, derivative->kernel);
    }

 private:
    // RawOperation exists only while lowering one source gate.  Masks express
    // source semantics conveniently; PrepareOperation projects them into the
    // strongly typed, mask-free hot-path kernels in circuit_tiling.h.
    enum class RawKernel : uint8_t {
        Pauli,
        PauliRotation,
        Phase,
        Hadamard,
        PairMatrix,
        Swap,
        ISwap,
        FSim,
        DenseMatrix,
        // A diagonal matrix on the operand block: `matrix` holds one factor
        // per entry of `target_offsets`.
        Diagonal,
        Depolarizing,
    };

    struct RawOperation {
        RawKernel kernel = RawKernel::DenseMatrix;
        index_t local_control_mask = 0;
        index_t external_control_mask = 0;
        index_t target_mask = 0;
        index_t target_mask_2 = 0;
        index_t pauli_x_mask = 0;
        index_t pauli_y_mask = 0;
        index_t pauli_z_mask = 0;
        index_t external_pauli_z_mask = 0;
        calc_type rotation_cos = 1;
        calc_type rotation_sin = 0;
        calc_type probability = 0;
        value_t phase = 1;
        std::array<value_t, 4> matrix2{value_t{1, 0}, value_t{0, 0}, value_t{0, 0}, value_t{1, 0}};
        size_t matrix_dim = 0;
        std::vector<index_t> target_offsets;
        std::vector<value_t> matrix;
    };

    [[nodiscard]] static auto IsImageSymmetric(const qbits_t& tile_qubits, qbit_t image_shift) -> bool {
        const auto half = tile_qubits.size() / 2;
        if (tile_qubits.size() != 2 * half) {
            return false;
        }
        for (size_t index = 0; index < half; ++index) {
            if (tile_qubits[index] >= image_shift || tile_qubits[index + half] != tile_qubits[index] + image_shift) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static auto BuildLayout(const CircuitTile& tile, const StateSpace& space) -> TileLayout {
        TileLayout layout;
        layout.qubits = tile.qubits;
        layout.dimension = size_t{1} << layout.qubits.size();
        layout.tile_count = space.dimension() >> layout.qubits.size();
        layout.state_offsets.resize(layout.dimension, 0);
        for (size_t bit = 0; bit < layout.qubits.size(); ++bit) {
            const auto offset = index_t{1} << layout.qubits[bit];
            const auto stride = size_t{1} << bit;
            for (size_t basis = stride; basis < layout.dimension; basis += 2 * stride) {
                for (size_t inner = 0; inner < stride; ++inner) {
                    layout.state_offsets[basis + inner] |= offset;
                }
            }
        }
        if (space.kind() == StateSpace::Kind::DensityMatrix && IsImageSymmetric(tile.qubits, space.circuit_qubits())) {
            layout.mirror_shift = space.circuit_qubits();
        }
        return layout;
    }

    static void ValidateTile(const CircuitTile& tile, const StateSpace& space) {
        if (tile.gates.empty()) {
            throw std::invalid_argument("A circuit tile cannot be empty.");
        }
        if (tile.qubits.size() >= std::numeric_limits<size_t>::digits) {
            throw std::invalid_argument("Circuit tile is too large for local indexing.");
        }
        if (!std::is_sorted(tile.qubits.begin(), tile.qubits.end())) {
            throw std::invalid_argument("Circuit tile qubits must be sorted.");
        }
        for (size_t index = 0; index < tile.qubits.size(); ++index) {
            const auto qubit = tile.qubits[index];
            if (qubit < 0 || qubit >= space.state_qubits() || (index != 0 && qubit == tile.qubits[index - 1])) {
                throw std::invalid_argument("Circuit tile qubits are invalid for the state.");
            }
        }
        if (std::find(tile.gates.begin(), tile.gates.end(), nullptr) != tile.gates.end()) {
            throw std::invalid_argument("A circuit tile cannot contain a null gate.");
        }
    }

    [[nodiscard]] static auto LowerGate(const BasicGate& gate, const qbits_t& tile_qubits,
                                        const parameter::ParameterResolver& parameter_resolver,
                                        const StateSpace& space) -> std::vector<Operation<value_t>> {
        const auto tile_dim = size_t{1} << tile_qubits.size();
        if (IsNoiseChannel(gate.GetID())) {
            return {PrepareOperation(BuildChannel(gate, tile_qubits, space), tile_dim)};
        }
        std::vector<Operation<value_t>> operations;
        for (const auto& image : space.Images()) {
            auto operation = PrepareOperation(BuildOperation(gate, tile_qubits, parameter_resolver, image), tile_dim);
            if (image.conjugated) {
                Conjugate(&operation);
            }
            operations.push_back(std::move(operation));
        }
        return operations;
    }

    [[nodiscard]] static auto ResolveScalar(const parameter::ParameterResolver& coefficient,
                                            const parameter::ParameterResolver& parameter_resolver) -> calc_type {
        return tensor::ops::cpu::to_vector<calc_type>(coefficient.Combination(parameter_resolver).const_value)[0];
    }

    static void RequireObjectCount(const BasicGate& gate, const qbits_t& objects, size_t expected) {
        if (objects.size() != expected) {
            throw std::invalid_argument(
                fmt::format("Gate {} requires {} object qubits, but got {}.", gate.GetID(), expected, objects.size()));
        }
    }

    [[nodiscard]] static auto LocalMask(const qbits_t& tile_qubits, qbit_t state_qubit) -> index_t {
        const auto position = std::lower_bound(tile_qubits.begin(), tile_qubits.end(), state_qubit);
        if (position == tile_qubits.end() || *position != state_qubit) {
            return 0;
        }
        return index_t{1} << std::distance(tile_qubits.begin(), position);
    }

    [[nodiscard]] static auto RequireLocalMask(const qbits_t& tile_qubits, qbit_t state_qubit) -> index_t {
        const auto local = LocalMask(tile_qubits, state_qubit);
        if (local == 0) {
            throw std::invalid_argument(
                fmt::format("State-index bit {} is mixed by a gate but is not part of its circuit tile.", state_qubit));
        }
        return local;
    }

    static void ConfigureControls(const BasicGate& gate, const qbits_t& tile_qubits, const QubitImage& image,
                                  RawOperation* operation) {
        for (auto control : gate.GetCtrlQubits()) {
            AddConditionQubit(control + image.offset, tile_qubits, operation);
        }
    }

    static void AddConditionQubit(qbit_t state_qubit, const qbits_t& tile_qubits, RawOperation* operation) {
        const auto local = LocalMask(tile_qubits, state_qubit);
        if (local != 0) {
            operation->local_control_mask |= local;
        } else {
            operation->external_control_mask |= index_t{1} << state_qubit;
        }
    }

    static void SetLocalOrExternalZ(qbit_t state_qubit, const qbits_t& tile_qubits, RawOperation* operation) {
        const auto local = LocalMask(tile_qubits, state_qubit);
        if (local != 0) {
            operation->pauli_z_mask |= local;
        } else {
            operation->external_pauli_z_mask |= index_t{1} << state_qubit;
        }
    }

    static void SetLocalPauli(qbit_t state_qubit, char pauli, const qbits_t& tile_qubits, RawOperation* operation) {
        switch (pauli) {
            case 'X':
                operation->pauli_x_mask |= RequireLocalMask(tile_qubits, state_qubit);
                break;
            case 'Y':
                operation->pauli_y_mask |= RequireLocalMask(tile_qubits, state_qubit);
                break;
            case 'Z':
                SetLocalOrExternalZ(state_qubit, tile_qubits, operation);
                break;
            default:
                assert(false && "Unexpected Pauli label");
                std::terminate();
        }
    }

    static void ConfigurePauliMask(const PauliMask& mask, const qbits_t& tile_qubits, const QubitImage& image,
                                   RawOperation* operation) {
        for (qbit_t qubit = 0; qubit < static_cast<qbit_t>(std::numeric_limits<index_t>::digits); ++qubit) {
            const auto global = index_t{1} << qubit;
            if ((mask.mask_x & global) != 0) {
                SetLocalPauli(qubit + image.offset, 'X', tile_qubits, operation);
            } else if ((mask.mask_y & global) != 0) {
                SetLocalPauli(qubit + image.offset, 'Y', tile_qubits, operation);
            } else if ((mask.mask_z & global) != 0) {
                SetLocalPauli(qubit + image.offset, 'Z', tile_qubits, operation);
            }
        }
    }

    [[nodiscard]] static auto MatrixFromTensor(const tensor::Matrix& matrix,
                                               size_t expected_dim) -> std::vector<value_t> {
        if (matrix.n_row != expected_dim || matrix.n_col != expected_dim) {
            throw std::invalid_argument(fmt::format("Gate matrix expects shape {}x{}, but got {}x{}.", expected_dim,
                                                    expected_dim, matrix.n_row, matrix.n_col));
        }
        auto nested = tensor::ops::cpu::to_vector<value_t>(matrix);
        std::vector<value_t> flat;
        flat.reserve(expected_dim * expected_dim);
        for (const auto& row : nested) {
            if (row.size() != expected_dim) {
                throw std::invalid_argument("Gate matrix rows have inconsistent sizes.");
            }
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return flat;
    }

    [[nodiscard]] static auto CustomMatrix(
        const BasicGate& gate, const parameter::ParameterResolver& parameter_resolver) -> std::vector<value_t> {
        const auto expected_dim = size_t{1} << gate.GetObjQubits().size();
        if (gate.GetID() == GateID::CUSTOM) {
            const auto* custom = static_cast<const CustomGate*>(&gate);
            if (!custom->Parameterized()) {
                return MatrixFromTensor(custom->GetBaseMatrix(), expected_dim);
            }
            const auto value = ResolveScalar(custom->GetCoeffs()[0], parameter_resolver);
            return MatrixFromTensor(custom->GetMatrixWrapper()(value), expected_dim);
        }
        const auto* custom = static_cast<const CustomTwoParamGate*>(&gate);
        if (!custom->Parameterized()) {
            return MatrixFromTensor(custom->GetBaseMatrix(), expected_dim);
        }
        const auto coefficients = custom->GetCoeffs();
        const auto first = ResolveScalar(coefficients[0], parameter_resolver);
        const auto second = ResolveScalar(coefficients[1], parameter_resolver);
        return MatrixFromTensor(custom->GetMatrixWrapper()(first, second), expected_dim);
    }

    // Local offsets of every basis state of `objects` placed in `image`.
    [[nodiscard]] static auto ObjectOffsets(const qbits_t& objects, const qbits_t& tile_qubits,
                                            const QubitImage& image) -> std::vector<index_t> {
        std::vector<index_t> offsets(size_t{1} << objects.size(), 0);
        for (size_t basis = 0; basis < offsets.size(); ++basis) {
            for (size_t bit = 0; bit < objects.size(); ++bit) {
                if ((basis & (size_t{1} << bit)) != 0) {
                    offsets[basis] |= RequireLocalMask(tile_qubits, objects[bit] + image.offset);
                }
            }
        }
        return offsets;
    }

    [[nodiscard]] static auto MatrixOperation(const BasicGate& gate, const qbits_t& tile_qubits,
                                              const QubitImage& image, std::vector<value_t> matrix) -> RawOperation {
        RawOperation operation;
        ConfigureControls(gate, tile_qubits, image, &operation);
        const auto objects = gate.GetObjQubits();
        operation.matrix_dim = size_t{1} << objects.size();
        if (matrix.size() != operation.matrix_dim * operation.matrix_dim) {
            throw std::invalid_argument("Matrix operation dimensions are inconsistent with its object qubits.");
        }
        if (objects.size() == 1) {
            operation.kernel = RawKernel::PairMatrix;
            operation.target_mask = RequireLocalMask(tile_qubits, objects.front() + image.offset);
            std::copy_n(matrix.begin(), 4, operation.matrix2.begin());
            operation.matrix_dim = 0;
            return operation;
        }

        operation.kernel = RawKernel::DenseMatrix;
        operation.matrix = std::move(matrix);
        operation.target_offsets = ObjectOffsets(objects, tile_qubits, image);
        return operation;
    }

    static void ConfigureRotation(const BasicGate& gate, const qbits_t& objects, const qbits_t& tile_qubits,
                                  const parameter::ParameterResolver& parameter_resolver, const QubitImage& image,
                                  RawOperation* operation) {
        const auto set_pauli = [&](size_t object_index, char pauli) {
            SetLocalPauli(objects[object_index] + image.offset, pauli, tile_qubits, operation);
        };
        switch (gate.GetID()) {
            case GateID::RX:
                RequireObjectCount(gate, objects, 1);
                set_pauli(0, 'X');
                break;
            case GateID::RY:
                RequireObjectCount(gate, objects, 1);
                set_pauli(0, 'Y');
                break;
            case GateID::RZ:
                RequireObjectCount(gate, objects, 1);
                set_pauli(0, 'Z');
                break;
            case GateID::Rxx:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'X');
                set_pauli(1, 'X');
                break;
            case GateID::Ryy:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'Y');
                set_pauli(1, 'Y');
                break;
            case GateID::Rzz:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'Z');
                set_pauli(1, 'Z');
                break;
            case GateID::Rxy:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'X');
                set_pauli(1, 'Y');
                break;
            case GateID::Rxz:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'X');
                set_pauli(1, 'Z');
                break;
            case GateID::Ryz:
                RequireObjectCount(gate, objects, 2);
                set_pauli(0, 'Y');
                set_pauli(1, 'Z');
                break;
            case GateID::RPS:
                ConfigurePauliMask(static_cast<const RotPauliString*>(&gate)->GetPauliString().GetPauliMask(),
                                   tile_qubits, image, operation);
                break;
            default:
                assert(false && "Unexpected Pauli rotation gate");
                std::terminate();
        }
        const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                         parameter_resolver);
        operation->rotation_cos = std::cos(angle / calc_type{2});
        operation->rotation_sin = std::sin(angle / calc_type{2});
    }

    [[nodiscard]] static auto BuildOperation(const BasicGate& gate, const qbits_t& tile_qubits,
                                             const parameter::ParameterResolver& parameter_resolver,
                                             const QubitImage& image) -> RawOperation {
        RawOperation operation;
        ConfigureControls(gate, tile_qubits, image, &operation);
        const auto objects = gate.GetObjQubits();
        const auto placed = [&image](qbit_t qubit) { return qubit + image.offset; };
        constexpr auto half = calc_type{0.5};
        switch (gate.GetID()) {
            case GateID::X:
            case GateID::Y:
            case GateID::Z: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::Pauli;
                const auto pauli = gate.GetID() == GateID::X ? 'X' : (gate.GetID() == GateID::Y ? 'Y' : 'Z');
                SetLocalPauli(placed(objects[0]), pauli, tile_qubits, &operation);
                break;
            }
            case GateID::PauliString:
                operation.kernel = RawKernel::Pauli;
                ConfigurePauliMask(static_cast<const PauliString*>(&gate)->GetPauliMask(), tile_qubits, image,
                                   &operation);
                break;
            case GateID::RX:
            case GateID::RY:
            case GateID::RZ:
            case GateID::Rxx:
            case GateID::Ryy:
            case GateID::Rzz:
            case GateID::Rxy:
            case GateID::Rxz:
            case GateID::Ryz:
            case GateID::RPS:
                operation.kernel = RawKernel::PauliRotation;
                ConfigureRotation(gate, objects, tile_qubits, parameter_resolver, image, &operation);
                break;
            case GateID::H:
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::Hadamard;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                break;
            case GateID::S:
            case GateID::Sdag:
            case GateID::T:
            case GateID::Tdag:
            case GateID::PS: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::Phase;
                operation.target_mask = LocalMask(tile_qubits, placed(objects[0]));
                if (operation.target_mask == 0) {
                    operation.external_control_mask |= index_t{1} << placed(objects[0]);
                }
                if (gate.GetID() == GateID::S) {
                    operation.phase = value_t{0, 1};
                } else if (gate.GetID() == GateID::Sdag) {
                    operation.phase = value_t{0, -1};
                } else if (gate.GetID() == GateID::T) {
                    operation.phase = value_t{M_SQRT1_2, M_SQRT1_2};
                } else if (gate.GetID() == GateID::Tdag) {
                    operation.phase = value_t{M_SQRT1_2, -M_SQRT1_2};
                } else {
                    const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                     parameter_resolver);
                    operation.phase = std::exp(value_t{0, angle});
                }
                break;
            }
            case GateID::GP: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::Phase;
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                operation.phase = std::exp(value_t{0, -angle});
                break;
            }
            case GateID::SX:
            case GateID::SXdag: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                const auto sign = gate.GetID() == GateID::SXdag ? calc_type{-1} : calc_type{1};
                const auto diagonal = value_t{half, sign * half};
                const auto off_diagonal = value_t{half, -sign * half};
                operation.matrix2 = {diagonal, off_diagonal, off_diagonal, diagonal};
                break;
            }
            case GateID::SWAP:
                RequireObjectCount(gate, objects, 2);
                operation.kernel = RawKernel::Swap;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                operation.target_mask_2 = RequireLocalMask(tile_qubits, placed(objects[1]));
                break;
            case GateID::ISWAP:
                RequireObjectCount(gate, objects, 2);
                operation.kernel = RawKernel::ISwap;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                operation.target_mask_2 = RequireLocalMask(tile_qubits, placed(objects[1]));
                operation.phase = static_cast<const ISWAPGate*>(&gate)->Daggered() ? value_t{0, -1} : value_t{0, 1};
                break;
            case GateID::SWAPalpha: {
                RequireObjectCount(gate, objects, 2);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                operation.target_mask_2 = RequireLocalMask(tile_qubits, placed(objects[1]));
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto exponential = std::exp(value_t{0, static_cast<calc_type>(M_PI) * angle});
                const auto diagonal = (value_t{1, 0} + exponential) / calc_type{2};
                const auto off_diagonal = (value_t{1, 0} - exponential) / calc_type{2};
                operation.matrix2 = {diagonal, off_diagonal, off_diagonal, diagonal};
                break;
            }
            case GateID::Givens: {
                RequireObjectCount(gate, objects, 2);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                operation.target_mask_2 = RequireLocalMask(tile_qubits, placed(objects[1]));
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto cosine = static_cast<calc_type>(std::cos(angle));
                const auto sine = static_cast<calc_type>(std::sin(angle));
                operation.matrix2 = {cosine, -sine, sine, cosine};
                break;
            }
            case GateID::FSim: {
                RequireObjectCount(gate, objects, 2);
                operation.kernel = RawKernel::FSim;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                operation.target_mask_2 = RequireLocalMask(tile_qubits, placed(objects[1]));
                const auto* fsim = static_cast<const FSim*>(&gate);
                const auto theta = ResolveScalar(fsim->GetTheta(), parameter_resolver);
                const auto phi = ResolveScalar(fsim->GetPhi(), parameter_resolver);
                const auto cosine = static_cast<calc_type>(std::cos(theta));
                const auto minus_i_sine = value_t{0, -static_cast<calc_type>(std::sin(theta))};
                operation.matrix2 = {cosine, minus_i_sine, minus_i_sine, cosine};
                operation.phase = std::exp(value_t{0, -phi});
                break;
            }
            case GateID::U3: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                const auto* u3 = static_cast<const U3*>(&gate);
                const auto theta = ResolveScalar(u3->GetTheta(), parameter_resolver);
                const auto phi = ResolveScalar(u3->GetPhi(), parameter_resolver);
                const auto lambda = ResolveScalar(u3->GetLambda(), parameter_resolver);
                const auto cosine = static_cast<calc_type>(std::cos(theta * half));
                const auto sine = static_cast<calc_type>(std::sin(theta * half));
                const auto exp_phi = std::exp(value_t{0, phi});
                const auto exp_lambda = std::exp(value_t{0, lambda});
                operation.matrix2 = {cosine, -exp_lambda * sine, exp_phi * sine, exp_phi * exp_lambda * cosine};
                break;
            }
            case GateID::Rn: {
                RequireObjectCount(gate, objects, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = RequireLocalMask(tile_qubits, placed(objects[0]));
                const auto* rn = static_cast<const Rn*>(&gate);
                const auto alpha = ResolveScalar(rn->GetAlpha(), parameter_resolver);
                const auto beta = ResolveScalar(rn->GetBeta(), parameter_resolver);
                const auto gamma = ResolveScalar(rn->GetGamma(), parameter_resolver);
                const auto norm = static_cast<calc_type>(std::sqrt(alpha * alpha + beta * beta + gamma * gamma));
                if (norm == 0) {
                    operation.matrix2 = {1, 0, 0, 1};
                } else {
                    const auto cosine = static_cast<calc_type>(std::cos(norm * half));
                    const auto sine_over_norm = static_cast<calc_type>(std::sin(norm * half) / norm);
                    operation.matrix2 = {value_t{cosine, -gamma * sine_over_norm},
                                         value_t{-beta * sine_over_norm, -alpha * sine_over_norm},
                                         value_t{beta * sine_over_norm, -alpha * sine_over_norm},
                                         value_t{cosine, gamma * sine_over_norm}};
                }
                break;
            }
            case GateID::CUSTOM:
            case GateID::CUSTOM_TWO_PARAM:
                return MatrixOperation(gate, tile_qubits, image, CustomMatrix(gate, parameter_resolver));
            default:
                throw std::invalid_argument(fmt::format("Gate {} cannot be lowered to a circuit tile.", gate.GetID()));
        }
        return operation;
    }

    [[nodiscard]] static auto BuildIntrinsicDerivatives(const BasicGate& gate, const qbits_t& tile_qubits,
                                                        const parameter::ParameterResolver& parameter_resolver,
                                                        const RawOperation& evolution,
                                                        const QubitImage& image) -> std::vector<RawOperation> {
        const auto objects = gate.GetObjQubits();
        switch (gate.GetID()) {
            case GateID::RX:
            case GateID::RY:
            case GateID::RZ:
            case GateID::Rxx:
            case GateID::Ryy:
            case GateID::Rzz:
            case GateID::Rxy:
            case GateID::Rxz:
            case GateID::Ryz:
            case GateID::RPS: {
                auto derivative = evolution;
                const auto cosine = derivative.rotation_cos;
                const auto sine = derivative.rotation_sin;
                derivative.rotation_cos = -sine / calc_type{2};
                derivative.rotation_sin = cosine / calc_type{2};
                return {std::move(derivative)};
            }
            case GateID::PS: {
                auto derivative = evolution;
                derivative.phase *= value_t{0, 1};
                return {std::move(derivative)};
            }
            case GateID::GP: {
                auto derivative = evolution;
                derivative.phase *= value_t{0, -1};
                return {std::move(derivative)};
            }
            case GateID::SWAPalpha: {
                auto derivative = evolution;
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto exponential = std::exp(value_t{0, static_cast<calc_type>(M_PI) * angle});
                const auto diagonal = value_t{0, static_cast<calc_type>(M_PI_2)} * exponential;
                derivative.matrix2 = {diagonal, -diagonal, -diagonal, diagonal};
                return {std::move(derivative)};
            }
            case GateID::Givens: {
                auto derivative = evolution;
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto cosine = static_cast<calc_type>(std::cos(angle));
                const auto minus_sine = static_cast<calc_type>(-std::sin(angle));
                derivative.matrix2 = {minus_sine, -cosine, cosine, minus_sine};
                return {std::move(derivative)};
            }
            case GateID::U3: {
                const auto* u3 = static_cast<const U3*>(&gate);
                const auto theta = u3->GetTheta().Combination(parameter_resolver).const_value;
                const auto phi = u3->GetPhi().Combination(parameter_resolver).const_value;
                const auto lambda = u3->GetLambda().Combination(parameter_resolver).const_value;
                return {
                    MatrixOperation(gate, tile_qubits, image,
                                    MatrixFromTensor(U3DiffThetaMatrix(theta, phi, lambda), 2)),
                    MatrixOperation(gate, tile_qubits, image, MatrixFromTensor(U3DiffPhiMatrix(theta, phi, lambda), 2)),
                    MatrixOperation(gate, tile_qubits, image,
                                    MatrixFromTensor(U3DiffLambdaMatrix(theta, phi, lambda), 2))};
            }
            case GateID::Rn: {
                const auto* rn = static_cast<const Rn*>(&gate);
                const auto alpha = rn->GetAlpha().Combination(parameter_resolver).const_value;
                const auto beta = rn->GetBeta().Combination(parameter_resolver).const_value;
                const auto gamma = rn->GetGamma().Combination(parameter_resolver).const_value;
                return {MatrixOperation(gate, tile_qubits, image,
                                        MatrixFromTensor(RnDiffAlphaMatrix(alpha, beta, gamma), 2)),
                        MatrixOperation(gate, tile_qubits, image,
                                        MatrixFromTensor(RnDiffBetaMatrix(alpha, beta, gamma), 2)),
                        MatrixOperation(gate, tile_qubits, image,
                                        MatrixFromTensor(RnDiffGammaMatrix(alpha, beta, gamma), 2))};
            }
            case GateID::FSim: {
                RequireObjectCount(gate, objects, 2);
                const auto* fsim = static_cast<const FSim*>(&gate);
                const auto theta = ResolveScalar(fsim->GetTheta(), parameter_resolver);
                const auto phi = ResolveScalar(fsim->GetPhi(), parameter_resolver);

                auto theta_derivative = evolution;
                theta_derivative.kernel = RawKernel::PairMatrix;
                const auto minus_sine = static_cast<calc_type>(-std::sin(theta));
                const auto minus_i_cosine = value_t{0, -static_cast<calc_type>(std::cos(theta))};
                theta_derivative.matrix2 = {minus_sine, minus_i_cosine, minus_i_cosine, minus_sine};

                RawOperation phi_derivative;
                phi_derivative.kernel = RawKernel::Phase;
                ConfigureControls(gate, tile_qubits, image, &phi_derivative);
                AddConditionQubit(objects[0] + image.offset, tile_qubits, &phi_derivative);
                AddConditionQubit(objects[1] + image.offset, tile_qubits, &phi_derivative);
                phi_derivative.phase = value_t{0, -1} * std::exp(value_t{0, -phi});
                return {std::move(theta_derivative), std::move(phi_derivative)};
            }
            case GateID::CUSTOM: {
                const auto* custom = static_cast<const CustomGate*>(&gate);
                const auto value = ResolveScalar(custom->GetCoeffs()[0], parameter_resolver);
                const auto expected_dim = size_t{1} << objects.size();
                return {MatrixOperation(gate, tile_qubits, image,
                                        MatrixFromTensor(custom->GetDiffMatrixWrapper()(value), expected_dim))};
            }
            case GateID::CUSTOM_TWO_PARAM: {
                const auto* custom = static_cast<const CustomTwoParamGate*>(&gate);
                const auto coefficients = custom->GetCoeffs();
                const auto first = ResolveScalar(coefficients[0], parameter_resolver);
                const auto second = ResolveScalar(coefficients[1], parameter_resolver);
                const auto expected_dim = size_t{1} << objects.size();
                return {
                    MatrixOperation(gate, tile_qubits, image,
                                    MatrixFromTensor(custom->GetDiffMatrix1Wrapper()(first, second), expected_dim)),
                    MatrixOperation(gate, tile_qubits, image,
                                    MatrixFromTensor(custom->GetDiffMatrix2Wrapper()(first, second), expected_dim))};
            }
            default:
                throw std::invalid_argument(
                    fmt::format("Gradient of gate {} cannot be lowered to a circuit tile.", gate.GetID()));
        }
    }

    [[nodiscard]] static auto BuildGradientTargets(const BasicGate& gate, const MST<size_t>& parameter_map,
                                                   size_t intrinsic_count)
        -> std::vector<std::vector<GradientTarget<value_t>>> {
        const auto& [title, jacobian] = static_cast<const Parameterizable*>(&gate)->GetJacobi();
        if (jacobian.n_row != intrinsic_count || jacobian.n_col != title.size()) {
            throw std::logic_error("Gate Jacobian dimensions do not match its intrinsic derivatives.");
        }
        const auto values = tensor::ops::cpu::to_vector<value_t>(jacobian);
        if (values.size() != intrinsic_count) {
            throw std::logic_error("Gate Jacobian rows are inconsistent.");
        }

        std::vector<std::vector<GradientTarget<value_t>>> targets(intrinsic_count);
        for (const auto& [name, column] : title) {
            const auto output = parameter_map.find(name);
            if (output == parameter_map.end()) {
                throw std::invalid_argument(
                    fmt::format("Gradient parameter '{}' is missing from the output map.", name));
            }
            if (output->second >= parameter_map.size() || column >= title.size()) {
                throw std::logic_error("Gradient parameter indices are inconsistent.");
            }
            for (size_t intrinsic = 0; intrinsic < intrinsic_count; ++intrinsic) {
                if (values[intrinsic].size() != title.size()) {
                    throw std::logic_error("Gate Jacobian columns are inconsistent.");
                }
                const auto coefficient = values[intrinsic][column];
                if (coefficient != value_t{0, 0}) {
                    targets[intrinsic].push_back({1 + output->second, coefficient});
                }
            }
        }
        return targets;
    }

    // Noise channels
    // ------------------------------------------------------------------------
    // A channel is lowered to the exact superoperator of its Kraus form on the
    // row-major (row, column) operand block of a density matrix.  Every
    // parameter is validated here, before any amplitude is touched: a channel
    // outside its physical range would otherwise poison the state silently.

    static void RequireChannel(bool condition, const BasicGate& gate, const char* requirement) {
        if (!condition) {
            throw std::invalid_argument(fmt::format("Channel {} requires {}.", gate.GetID(), requirement));
        }
    }

    [[nodiscard]] static auto RequireUnitInterval(double value, const BasicGate& gate,
                                                  const char* requirement) -> calc_type {
        RequireChannel(IsFinite(value) && value >= 0 && value <= 1, gate, requirement);
        return static_cast<calc_type>(value);
    }

    [[nodiscard]] static auto ConjugateTranspose(const std::vector<value_t>& matrix,
                                                 size_t dim) -> std::vector<value_t> {
        std::vector<value_t> result(matrix.size());
        for (size_t row = 0; row < dim; ++row) {
            for (size_t column = 0; column < dim; ++column) {
                result[row * dim + column] = std::conj(matrix[column * dim + row]);
            }
        }
        return result;
    }

    // Adds weight * (K rho K^dagger) to the superoperator on a 2^m-dimensional
    // operand: S[(r', c'), (r, c)] += weight * K[r'][r] conj(K[c'][c]).
    static void AccumulateKrausTerm(calc_type weight, const std::vector<value_t>& kraus, size_t dim,
                                    std::vector<value_t>* superoperator) {
        assert(kraus.size() == dim * dim && superoperator->size() == dim * dim * dim * dim);
        const auto super_dim = dim * dim;
        for (size_t row_out = 0; row_out < dim; ++row_out) {
            for (size_t column_out = 0; column_out < dim; ++column_out) {
                auto* out = superoperator->data() + (row_out * dim + column_out) * super_dim;
                for (size_t row_in = 0; row_in < dim; ++row_in) {
                    const auto left = weight * kraus[row_out * dim + row_in];
                    for (size_t column_in = 0; column_in < dim; ++column_in) {
                        out[row_in * dim + column_in] += left * std::conj(kraus[column_out * dim + column_in]);
                    }
                }
            }
        }
    }

    // Kraus operators of a user-defined channel, validated as a complete set:
    // sum_k K_k^dagger K_k = I.  The check runs in double precision so that
    // it does not depend on the simulator's value type.
    [[nodiscard]] static auto ValidatedKrausOperators(const KrausChannel& gate,
                                                      size_t dim) -> std::vector<std::vector<value_t>> {
        const auto& operators = gate.GetKrausOperatorSet();
        RequireChannel(!operators.empty(), gate, "at least one Kraus operator");
        std::vector<std::complex<double>> completeness(dim * dim, 0);
        std::vector<std::vector<value_t>> lowered;
        lowered.reserve(operators.size());
        for (const auto& matrix : operators) {
            RequireChannel(matrix.n_row == dim && matrix.n_col == dim, gate,
                           "square Kraus operators over its object qubits");
            const auto rows = tensor::ops::cpu::to_vector<std::complex<double>>(matrix);
            std::vector<value_t> flat;
            flat.reserve(dim * dim);
            for (const auto& row : rows) {
                RequireChannel(row.size() == dim, gate, "square Kraus operators over its object qubits");
                for (const auto entry : row) {
                    RequireChannel(IsFinite(entry.real()) && IsFinite(entry.imag()), gate,
                                   "finite Kraus operator entries");
                    flat.push_back(value_t{static_cast<calc_type>(entry.real()), static_cast<calc_type>(entry.imag())});
                }
            }
            for (size_t k = 0; k < dim; ++k) {
                for (size_t row = 0; row < dim; ++row) {
                    for (size_t column = 0; column < dim; ++column) {
                        completeness[row * dim + column] += std::conj(rows[k][row]) * rows[k][column];
                    }
                }
            }
            lowered.push_back(std::move(flat));
        }
        constexpr double tolerance = 1e-6;
        for (size_t row = 0; row < dim; ++row) {
            for (size_t column = 0; column < dim; ++column) {
                const auto expected = row == column ? 1.0 : 0.0;
                RequireChannel(std::abs(completeness[row * dim + column] - expected) <= tolerance, gate,
                               "Kraus operators satisfying sum_k K_k^dagger K_k = I");
            }
        }
        return lowered;
    }

    // The superoperator of a channel on the row-major (row, column) operand
    // block: S[(r', c'), (r, c)] = sum_k K_k[r'][r] conj(K_k[c'][c]).
    [[nodiscard]] static auto ChannelSuperoperator(const BasicGate& gate) -> std::vector<value_t> {
        const auto objects = gate.GetObjQubits();
        const auto dim = size_t{1} << objects.size();
        std::vector<value_t> superoperator(dim * dim * dim * dim, 0);
        switch (gate.GetID()) {
            case GateID::AD: {
                RequireObjectCount(gate, objects, 1);
                const auto* channel = static_cast<const AmplitudeDampingChannel*>(&gate);
                const auto gamma = RequireUnitInterval(channel->GetDampingCoeff(), gate,
                                                       "a damping coefficient in [0, 1]");
                std::vector<value_t> decay{0, std::sqrt(gamma), 0, 0};
                if (channel->Daggered()) {
                    decay = ConjugateTranspose(decay, 2);
                }
                AccumulateKrausTerm(1, {1, 0, 0, std::sqrt(1 - gamma)}, dim, &superoperator);
                AccumulateKrausTerm(1, decay, dim, &superoperator);
                break;
            }
            case GateID::PD: {
                RequireObjectCount(gate, objects, 1);
                const auto gamma = RequireUnitInterval(
                    static_cast<const PhaseDampingChannel*>(&gate)->GetDampingCoeff(), gate,
                    "a damping coefficient in [0, 1]");
                AccumulateKrausTerm(1, {1, 0, 0, std::sqrt(1 - gamma)}, dim, &superoperator);
                AccumulateKrausTerm(1, {0, 0, 0, std::sqrt(gamma)}, dim, &superoperator);
                break;
            }
            case GateID::PL: {
                RequireObjectCount(gate, objects, 1);
                const auto probabilities = static_cast<const PauliChannel*>(&gate)->GetProbs();
                const auto x = RequireUnitInterval(probabilities[0], gate, "Pauli probabilities in [0, 1]");
                const auto y = RequireUnitInterval(probabilities[1], gate, "Pauli probabilities in [0, 1]");
                const auto z = RequireUnitInterval(probabilities[2], gate, "Pauli probabilities in [0, 1]");
                RequireChannel(probabilities[0] + probabilities[1] + probabilities[2] <= 1, gate,
                               "Pauli probabilities summing to at most 1");
                // rho -> sum_k p_k P_k rho P_k is linear in the probabilities,
                // so the Pauli matrices enter unscaled.
                AccumulateKrausTerm(x, {0, 1, 1, 0}, dim, &superoperator);
                AccumulateKrausTerm(y, {0, value_t{0, -1}, value_t{0, 1}, 0}, dim, &superoperator);
                AccumulateKrausTerm(z, {1, 0, 0, -1}, dim, &superoperator);
                AccumulateKrausTerm(1 - (x + y + z), {1, 0, 0, 1}, dim, &superoperator);
                break;
            }
            case GateID::KRAUS:
                for (const auto& kraus : ValidatedKrausOperators(*static_cast<const KrausChannel*>(&gate), dim)) {
                    AccumulateKrausTerm(1, kraus, dim, &superoperator);
                }
                break;
            case GateID::TR: {
                RequireObjectCount(gate, objects, 1);
                const auto* channel = static_cast<const ThermalRelaxationChannel*>(&gate);
                const auto t1 = channel->GetT1();
                const auto t2 = channel->GetT2();
                const auto gate_time = channel->GetGateTime();
                RequireChannel(IsFinite(t1) && t1 > 0 && IsFinite(t2) && t2 > 0, gate, "positive, finite T1 and T2");
                RequireChannel(IsFinite(gate_time) && gate_time >= 0, gate, "a non-negative, finite gate time");
                RequireChannel(t2 <= 2 * t1, gate, "T2 <= 2 * T1");
                const auto e1 = static_cast<calc_type>(std::exp(-gate_time / t1));
                const auto e2 = static_cast<calc_type>(std::exp(-gate_time / t2));
                // rho00 += (1 - e1) rho11, rho01 *= e2, rho10 *= e2, rho11 *= e1.
                superoperator[0 * 4 + 0] = 1;
                superoperator[0 * 4 + 3] = 1 - e1;
                superoperator[1 * 4 + 1] = e2;
                superoperator[2 * 4 + 2] = e2;
                superoperator[3 * 4 + 3] = e1;
                break;
            }
            default:
                throw std::invalid_argument(fmt::format("Gate {} is not a Kraus channel.", gate.GetID()));
        }
        return superoperator;
    }

    [[nodiscard]] static auto IsDiagonal(const std::vector<value_t>& matrix, size_t dim) -> bool {
        for (size_t row = 0; row < dim; ++row) {
            for (size_t column = 0; column < dim; ++column) {
                if (row != column && matrix[row * dim + column] != value_t{0, 0}) {
                    return false;
                }
            }
        }
        return true;
    }

    // Local offsets of the (row, column) operand block of `objects`, ordered
    // row-major to match the superoperator index convention.
    [[nodiscard]] static auto OperandBlockOffsets(const qbits_t& objects, const qbits_t& tile_qubits,
                                                  const StateSpace& space) -> std::vector<index_t> {
        const auto row_offsets = ObjectOffsets(objects, tile_qubits, space.RowImage());
        const auto column_offsets = ObjectOffsets(objects, tile_qubits, space.ColumnImage());
        std::vector<index_t> offsets;
        offsets.reserve(row_offsets.size() * column_offsets.size());
        for (auto row : row_offsets) {
            for (auto column : column_offsets) {
                offsets.push_back(row | column);
            }
        }
        return offsets;
    }

    [[nodiscard]] static auto BuildChannel(const BasicGate& gate, const qbits_t& tile_qubits,
                                           const StateSpace& space) -> RawOperation {
        if (space.kind() != StateSpace::Kind::DensityMatrix) {
            throw std::invalid_argument(
                fmt::format("Noise channel {} can only be lowered onto a density matrix.", gate.GetID()));
        }
        if (gate.GetID() == GateID::GPL) {
            throw std::invalid_argument("Grouped Pauli channels must be expanded before lowering.");
        }
        RequireChannel(gate.GetCtrlQubits().empty(), gate, "no control qubits");
        const auto objects = gate.GetObjQubits();
        RequireChannel(!objects.empty(), gate, "object qubits");
        RawOperation operation;
        operation.target_offsets = OperandBlockOffsets(objects, tile_qubits, space);
        if (gate.GetID() == GateID::DEP) {
            // rho -> (1 - p) rho + p Tr_Q(rho) (x) I / 2^m is completely
            // positive up to p = 4^m / (4^m - 1), where it inverts the Bloch
            // sphere.
            const auto probability = static_cast<const DepolarizingChannel*>(&gate)->GetProb();
            const auto pauli_count = std::ldexp(1.0, 2 * static_cast<int>(objects.size()));
            RequireChannel(IsFinite(probability) && probability >= 0 && probability <= pauli_count / (pauli_count - 1),
                           gate, "a probability in [0, 4^m / (4^m - 1)]");
            operation.kernel = RawKernel::Depolarizing;
            operation.matrix_dim = size_t{1} << objects.size();
            operation.probability = static_cast<calc_type>(probability);
            return operation;
        }
        operation.matrix_dim = operation.target_offsets.size();
        operation.matrix = ChannelSuperoperator(gate);
        if (IsDiagonal(operation.matrix, operation.matrix_dim)) {
            // Dephasing-type channels scale each block entry independently;
            // the diagonal kernel folds into neighbouring diagonal gates.
            std::vector<value_t> diagonal(operation.matrix_dim);
            for (size_t entry = 0; entry < operation.matrix_dim; ++entry) {
                diagonal[entry] = operation.matrix[entry * operation.matrix_dim + entry];
            }
            operation.kernel = RawKernel::Diagonal;
            operation.matrix = std::move(diagonal);
            return operation;
        }
        operation.kernel = RawKernel::DenseMatrix;
        return operation;
    }

    // Lowering to hot-path kernels
    // ------------------------------------------------------------------------
    //
    // The loops in this section run once per local basis (up to
    // 2^kMaxTileQubits times) for every lowered gate, and the builder is
    // instantiated inside the pybind modules, whose translation units have
    // exhausted GCC's per-unit inline budget.  A helper that is not trivially
    // small is therefore not reliably inlined there: whether it is depends on
    // unrelated code elsewhere in the module, and a lost inlining shows up as
    // a per-gate constant that dominates small-state circuits (measured: +15%
    // on a 4-qubit density matrix when a per-basis phase helper fell out).
    // Keep these loop bodies self-contained; if a per-basis helper is
    // unavoidable, mark it MQ_ALWAYS_INLINE.

    // The local Pauli string maps |basis> to phase(basis) |basis ^ flips> with
    // phase(basis) = i^(#Y) (-1)^parity(basis & (Y | Z)).  The bases are
    // visited in Gray code order, in which consecutive bases differ in one
    // bit, so the sign follows incrementally; flipping the X and Y bits
    // toggles it exactly when #Y is odd, which relates the two phases of a
    // pair.  The loop body is deliberately self-contained: it runs once per
    // local basis for every lowered Pauli gate.
    template <typename pauli_kernel_t>
    static void PreparePauliEntries(const RawOperation& raw, size_t tile_dim, pauli_kernel_t* kernel) {
        constexpr std::array<value_t, 4> powers_of_i{value_t{1, 0}, value_t{0, 1}, value_t{-1, 0}, value_t{0, -1}};
        const auto y_count = CountOne(static_cast<std::uint64_t>(raw.pauli_y_mask));
        const auto sign_mask = raw.pauli_y_mask | raw.pauli_z_mask;
        const auto flip_mask = raw.pauli_x_mask | raw.pauli_y_mask;
        const auto partner_sign = (y_count & 1U) != 0 ? calc_type{-1} : calc_type{1};
        if (flip_mask == 0) {
            kernel->scales.reserve(tile_dim);
        } else {
            kernel->pairs.reserve(tile_dim / 2);
        }

        auto phase = powers_of_i[y_count % 4];
        size_t basis = 0;
        for (size_t step = 0; step < tile_dim; ++step) {
            if ((basis & raw.local_control_mask) == raw.local_control_mask) {
                if (flip_mask == 0) {
                    kernel->scales.push_back({basis, phase});
                } else if (const auto partner = basis ^ flip_mask; basis < partner) {
                    kernel->pairs.push_back({{basis, partner}, phase * partner_sign, phase});
                }
            }
            const auto toggled = (step + 1) & (~step);  // the lowest set bit of step + 1
            basis ^= toggled;
            if ((toggled & sign_mask) != 0) {
                phase = -phase;
            }
        }
    }

    [[nodiscard]] static auto PreparePauli(RawOperation raw, size_t tile_dim) -> Operation<value_t> {
        PauliKernel<value_t> kernel;
        kernel.external_z_mask = raw.external_pauli_z_mask;
        PreparePauliEntries(raw, tile_dim, &kernel);
        Operation<value_t> operation;
        operation.external_control_mask = raw.external_control_mask;
        if (kernel.external_z_mask == 0 && kernel.pairs.empty()) {
            operation.kernel = DiagonalKernel<value_t>{std::move(kernel.scales)};
        } else {
            operation.kernel = std::move(kernel);
        }
        return operation;
    }

    [[nodiscard]] static auto PreparePauliRotation(RawOperation raw, size_t tile_dim) -> Operation<value_t> {
        PauliRotationKernel<value_t> kernel;
        kernel.external_z_mask = raw.external_pauli_z_mask;
        kernel.cosine = raw.rotation_cos;
        kernel.sine = raw.rotation_sin;
        PreparePauliEntries(raw, tile_dim, &kernel);
        if (kernel.external_z_mask == 0) {
            const auto minus_i_sine = value_t{0, -kernel.sine};
            for (auto& scale : kernel.scales) {
                scale.factor = value_t{kernel.cosine, 0} + minus_i_sine * scale.factor;
            }
            for (auto& pair : kernel.pairs) {
                pair.lhs_from_rhs *= minus_i_sine;
                pair.rhs_from_lhs *= minus_i_sine;
            }
            kernel.coefficients_prepared = true;
        }

        Operation<value_t> operation;
        operation.external_control_mask = raw.external_control_mask;
        if (kernel.coefficients_prepared && kernel.pairs.empty()) {
            operation.kernel = DiagonalKernel<value_t>{std::move(kernel.scales)};
        } else {
            operation.kernel = std::move(kernel);
        }
        return operation;
    }

    [[nodiscard]] static auto PrepareDiagonal(RawOperation raw, size_t tile_dim) -> Operation<value_t> {
        DiagonalKernel<value_t> kernel;
        kernel.scales.reserve(tile_dim);
        for (size_t basis = 0; basis < tile_dim; ++basis) {
            const auto controls_match = (basis & raw.local_control_mask) == static_cast<size_t>(raw.local_control_mask);
            const auto target_matches = raw.target_mask == 0 || (basis & raw.target_mask) != 0;
            if (controls_match && target_matches) {
                kernel.scales.push_back({basis, raw.phase});
            }
        }
        return {raw.external_control_mask, std::move(kernel)};
    }

    [[nodiscard]] static auto PreparePairs(const RawOperation& raw, size_t tile_dim) -> std::vector<IndexPair> {
        assert(raw.target_mask != 0);
        const auto controls_match = [&raw](size_t basis) {
            return (basis & raw.local_control_mask) == raw.local_control_mask;
        };
        std::vector<IndexPair> pairs;
        pairs.reserve(tile_dim / 2);
        if (raw.target_mask_2 == 0) {
            for (size_t basis = 0; basis < tile_dim; ++basis) {
                if (controls_match(basis) && (basis & raw.target_mask) == 0) {
                    pairs.push_back({basis, basis | raw.target_mask});
                }
            }
            return pairs;
        }

        assert(raw.target_mask != raw.target_mask_2);
        const auto object_mask = raw.target_mask | raw.target_mask_2;
        for (size_t basis = 0; basis < tile_dim; ++basis) {
            if (controls_match(basis) && (basis & object_mask) == 0) {
                pairs.push_back({basis | raw.target_mask, basis | raw.target_mask_2});
            }
        }
        return pairs;
    }

    // Visits every local basis of every operand block that satisfies the
    // local controls, as (tile basis, position within the block).
    template <typename visit_t>
    static void ForEachOperandBlock(const RawOperation& raw, size_t tile_dim, const visit_t& visit) {
        index_t object_mask = 0;
        for (auto offset : raw.target_offsets) {
            object_mask |= offset;
        }
        for (size_t basis = 0; basis < tile_dim; ++basis) {
            if ((basis & raw.local_control_mask) != raw.local_control_mask || (basis & object_mask) != 0) {
                continue;
            }
            for (size_t entry = 0; entry < raw.target_offsets.size(); ++entry) {
                visit(basis | raw.target_offsets[entry], entry);
            }
        }
    }

    [[nodiscard]] static auto PrepareDense(RawOperation raw, size_t tile_dim) -> DenseMatrixKernel<value_t> {
        assert(raw.matrix_dim != 0 && raw.target_offsets.size() == raw.matrix_dim);
        assert(raw.matrix.size() == raw.matrix_dim * raw.matrix_dim);
        DenseMatrixKernel<value_t> kernel;
        kernel.matrix_dim = raw.matrix_dim;
        kernel.matrix = std::move(raw.matrix);
        ForEachOperandBlock(raw, tile_dim, [&kernel](size_t index, size_t) { kernel.indices.push_back(index); });
        return kernel;
    }

    [[nodiscard]] static auto PrepareDiagonalTable(const RawOperation& raw,
                                                   size_t tile_dim) -> DiagonalKernel<value_t> {
        assert(raw.matrix.size() == raw.target_offsets.size());
        DiagonalKernel<value_t> kernel;
        ForEachOperandBlock(raw, tile_dim, [&raw, &kernel](size_t index, size_t entry) {
            if (raw.matrix[entry] != value_t{1, 0}) {
                kernel.scales.push_back({index, raw.matrix[entry]});
            }
        });
        return kernel;
    }

    [[nodiscard]] static auto PrepareDepolarizing(RawOperation raw, size_t tile_dim) -> DepolarizingKernel<value_t> {
        assert(raw.matrix_dim != 0 && raw.target_offsets.size() == raw.matrix_dim * raw.matrix_dim);
        DepolarizingKernel<value_t> kernel;
        kernel.local_dim = raw.matrix_dim;
        kernel.keep = 1 - raw.probability;
        kernel.mix = raw.probability / static_cast<calc_type>(raw.matrix_dim);
        ForEachOperandBlock(raw, tile_dim, [&kernel](size_t index, size_t) { kernel.indices.push_back(index); });
        return kernel;
    }

    [[nodiscard]] static auto PrepareOperation(RawOperation raw, size_t tile_dim) -> Operation<value_t> {
        switch (raw.kernel) {
            case RawKernel::Pauli:
                return PreparePauli(std::move(raw), tile_dim);
            case RawKernel::PauliRotation:
                return PreparePauliRotation(std::move(raw), tile_dim);
            case RawKernel::Phase:
                return PrepareDiagonal(std::move(raw), tile_dim);
            case RawKernel::Hadamard:
                return {raw.external_control_mask, HadamardKernel{PreparePairs(raw, tile_dim)}};
            case RawKernel::PairMatrix:
                return {raw.external_control_mask, PairMatrixKernel<value_t>{raw.matrix2, PreparePairs(raw, tile_dim)}};
            case RawKernel::Swap:
                return {raw.external_control_mask, SwapKernel{PreparePairs(raw, tile_dim)}};
            case RawKernel::ISwap:
                return {raw.external_control_mask, ISwapKernel<value_t>{raw.phase, PreparePairs(raw, tile_dim)}};
            case RawKernel::FSim: {
                const auto pairs = PreparePairs(raw, tile_dim);
                FSimKernel<value_t> kernel;
                kernel.exchange = raw.matrix2;
                kernel.phase = raw.phase;
                kernel.indices.reserve(pairs.size());
                for (const auto& pair : pairs) {
                    kernel.indices.push_back({pair.lhs, pair.rhs, pair.lhs | raw.target_mask_2});
                }
                return {raw.external_control_mask, std::move(kernel)};
            }
            case RawKernel::DenseMatrix:
                return {raw.external_control_mask, PrepareDense(std::move(raw), tile_dim)};
            case RawKernel::Diagonal:
                return {raw.external_control_mask, PrepareDiagonalTable(raw, tile_dim)};
            case RawKernel::Depolarizing:
                return {raw.external_control_mask, PrepareDepolarizing(std::move(raw), tile_dim)};
        }
        assert(false && "Unknown raw circuit-tile kernel");
        std::terminate();
    }

    [[nodiscard]] static auto PrepareDerivative(RawOperation raw, size_t tile_dim) -> Derivative<value_t> {
        auto operation = PrepareOperation(std::move(raw), tile_dim);
        Derivative<value_t> derivative;
        derivative.external_control_mask = operation.external_control_mask;
        if (auto* kernel = std::get_if<PauliRotationKernel<value_t>>(&operation.kernel)) {
            derivative.kernel = std::move(*kernel);
        } else if (auto* kernel = std::get_if<DiagonalKernel<value_t>>(&operation.kernel)) {
            derivative.kernel = std::move(*kernel);
        } else if (auto* kernel = std::get_if<PairMatrixKernel<value_t>>(&operation.kernel)) {
            derivative.kernel = std::move(*kernel);
        } else if (auto* kernel = std::get_if<DenseMatrixKernel<value_t>>(&operation.kernel)) {
            derivative.kernel = std::move(*kernel);
        } else {
            assert(false && "Unsupported derivative circuit-tile kernel");
            std::terminate();
        }
        return derivative;
    }

    [[nodiscard]] static auto FoldDiagonalRuns(std::vector<Operation<value_t>> operations,
                                               size_t tile_dim) -> std::vector<Operation<value_t>> {
        std::vector<Operation<value_t>> folded;
        folded.reserve(operations.size());
        for (size_t operation_index = 0; operation_index < operations.size();) {
            const auto* diagonal = std::get_if<DiagonalKernel<value_t>>(&operations[operation_index].kernel);
            if (operations[operation_index].external_control_mask != 0 || diagonal == nullptr) {
                folded.push_back(std::move(operations[operation_index++]));
                continue;
            }

            std::vector<value_t> factors(tile_dim, value_t{1, 0});
            do {
                for (const auto& scale : diagonal->scales) {
                    factors[scale.basis] *= scale.factor;
                }
                ++operation_index;
                if (operation_index == operations.size() || operations[operation_index].external_control_mask != 0) {
                    break;
                }
                diagonal = std::get_if<DiagonalKernel<value_t>>(&operations[operation_index].kernel);
            } while (diagonal != nullptr);

            DiagonalKernel<value_t> combined;
            for (size_t basis = 0; basis < tile_dim; ++basis) {
                if (factors[basis] != value_t{1, 0}) {
                    combined.scales.push_back({basis, factors[basis]});
                }
            }
            if (!combined.scales.empty()) {
                folded.push_back({0, std::move(combined)});
            }
        }
        return folded;
    }

    [[nodiscard]] static auto WorkspaceSize(const Kernel<value_t>& kernel) -> size_t {
        if (const auto* dense = std::get_if<DenseMatrixKernel<value_t>>(&kernel); dense != nullptr) {
            return dense->matrix_dim;
        }
        return 0;
    }

    // The local bases a prepared kernel reads or writes.  Every other basis
    // is left unchanged by the kernel, which for a derivative operator means
    // it lies outside the operator's support.
    [[nodiscard]] static auto TouchedBases(const Kernel<value_t>& kernel, size_t tile_dim) -> std::vector<bool> {
        std::vector<bool> touched(tile_dim, false);
        std::visit([&touched](const auto& typed) { MarkTouched(typed, &touched); }, kernel);
        return touched;
    }

    template <typename pauli_kernel_t>
    static void MarkTouchedPauli(const pauli_kernel_t& kernel, std::vector<bool>* touched) {
        for (const auto& scale : kernel.scales) {
            (*touched)[scale.basis] = true;
        }
        MarkTouchedPairs(kernel.pairs, touched);
    }

    template <typename pair_t>
    static void MarkTouchedPairs(const std::vector<pair_t>& pairs, std::vector<bool>* touched) {
        for (const auto& pair : pairs) {
            (*touched)[pair.lhs] = true;
            (*touched)[pair.rhs] = true;
        }
    }

    static void MarkTouched(const PauliKernel<value_t>& kernel, std::vector<bool>* touched) {
        MarkTouchedPauli(kernel, touched);
    }

    static void MarkTouched(const PauliRotationKernel<value_t>& kernel, std::vector<bool>* touched) {
        MarkTouchedPauli(kernel, touched);
    }

    static void MarkTouched(const DiagonalKernel<value_t>& kernel, std::vector<bool>* touched) {
        for (const auto& scale : kernel.scales) {
            (*touched)[scale.basis] = true;
        }
    }

    static void MarkTouched(const HadamardKernel& kernel, std::vector<bool>* touched) {
        MarkTouchedPairs(kernel.pairs, touched);
    }

    static void MarkTouched(const PairMatrixKernel<value_t>& kernel, std::vector<bool>* touched) {
        MarkTouchedPairs(kernel.pairs, touched);
    }

    static void MarkTouched(const SwapKernel& kernel, std::vector<bool>* touched) {
        MarkTouchedPairs(kernel.pairs, touched);
    }

    static void MarkTouched(const ISwapKernel<value_t>& kernel, std::vector<bool>* touched) {
        MarkTouchedPairs(kernel.pairs, touched);
    }

    static void MarkTouched(const FSimKernel<value_t>& kernel, std::vector<bool>* touched) {
        for (const auto& indices : kernel.indices) {
            (*touched)[indices.lhs] = true;
            (*touched)[indices.rhs] = true;
            (*touched)[indices.both] = true;
        }
    }

    static void MarkTouched(const DenseMatrixKernel<value_t>& kernel, std::vector<bool>* touched) {
        for (const auto index : kernel.indices) {
            (*touched)[index] = true;
        }
    }

    static void MarkTouched(const DepolarizingKernel<value_t>& kernel, std::vector<bool>* touched) {
        for (const auto index : kernel.indices) {
            (*touched)[index] = true;
        }
    }

    [[nodiscard]] static auto WorkspaceSize(const DerivativeKernel<value_t>& kernel) -> size_t {
        if (const auto* dense = std::get_if<DenseMatrixKernel<value_t>>(&kernel); dense != nullptr) {
            return dense->matrix_dim;
        }
        return 0;
    }

    // Conjugation and adjoint of lowered kernels
    // ------------------------------------------------------------------------
    // Every kernel is a linear map with explicit complex coefficients over a
    // real index structure, so conjugation conjugates the coefficients and
    // the adjoint additionally transposes the index structure.

    template <typename entries_t>
    static void ConjugateEntries(entries_t* entries) {
        for (auto& entry : *entries) {
            entry.factor = std::conj(entry.factor);
        }
    }

    static void ConjugatePairs(std::vector<PauliPair<value_t>>* pairs) {
        for (auto& pair : *pairs) {
            pair.lhs_from_rhs = std::conj(pair.lhs_from_rhs);
            pair.rhs_from_lhs = std::conj(pair.rhs_from_lhs);
        }
    }

    static void TransposePairs(std::vector<PauliPair<value_t>>* pairs) {
        for (auto& pair : *pairs) {
            std::swap(pair.lhs_from_rhs, pair.rhs_from_lhs);
        }
    }

    static void ConjugateKernel(PauliKernel<value_t>* kernel) {
        ConjugateEntries(&kernel->scales);
        ConjugatePairs(&kernel->pairs);
    }

    static void ConjugateKernel(PauliRotationKernel<value_t>* kernel) {
        // cos I - i sin P conjugates to cos I + i sin conj(P).
        kernel->sine = -kernel->sine;
        ConjugateEntries(&kernel->scales);
        ConjugatePairs(&kernel->pairs);
    }

    static void ConjugateKernel(DiagonalKernel<value_t>* kernel) {
        ConjugateEntries(&kernel->scales);
    }

    static void ConjugateKernel(HadamardKernel*) {
    }

    static void ConjugateKernel(PairMatrixKernel<value_t>* kernel) {
        for (auto& entry : kernel->matrix) {
            entry = std::conj(entry);
        }
    }

    static void ConjugateKernel(SwapKernel*) {
    }

    static void ConjugateKernel(ISwapKernel<value_t>* kernel) {
        kernel->phase = std::conj(kernel->phase);
    }

    static void ConjugateKernel(FSimKernel<value_t>* kernel) {
        for (auto& entry : kernel->exchange) {
            entry = std::conj(entry);
        }
        kernel->phase = std::conj(kernel->phase);
    }

    static void ConjugateKernel(DenseMatrixKernel<value_t>* kernel) {
        for (auto& entry : kernel->matrix) {
            entry = std::conj(entry);
        }
    }

    static void ConjugateKernel(DepolarizingKernel<value_t>*) {
    }

    static void AdjointKernel(PauliKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
        TransposePairs(&kernel->pairs);
    }

    static void AdjointKernel(PauliRotationKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
        TransposePairs(&kernel->pairs);
    }

    static void AdjointKernel(DiagonalKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
    }

    static void AdjointKernel(HadamardKernel*) {
    }

    static void AdjointKernel(PairMatrixKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
        std::swap(kernel->matrix[1], kernel->matrix[2]);
    }

    static void AdjointKernel(SwapKernel*) {
    }

    static void AdjointKernel(ISwapKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
    }

    static void AdjointKernel(FSimKernel<value_t>* kernel) {
        ConjugateKernel(kernel);
        std::swap(kernel->exchange[1], kernel->exchange[2]);
    }

    static void AdjointKernel(DenseMatrixKernel<value_t>* kernel) {
        kernel->matrix = ConjugateTranspose(kernel->matrix, kernel->matrix_dim);
    }

    static void AdjointKernel(DepolarizingKernel<value_t>*) {
    }
};
}  // namespace mindquantum::sim::circuit_tiling

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_BUILDER_H_
