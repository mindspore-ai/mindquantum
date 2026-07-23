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

#ifndef INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_BUILDER_H_
#define INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_BUILDER_H_

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
#include <vector>

#include <fmt/format.h>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "simulator/vector/circuit_tiling.h"

namespace mindquantum::sim::vector::detail::circuit_tiling {
template <typename value_t>
class Builder {
 public:
    using calc_type = typename value_t::value_type;

    [[nodiscard]] static auto Build(const CircuitTile& tile, const parameter::ParameterResolver& parameter_resolver,
                                    index_t state_dim) -> PreparedTile<value_t> {
        ValidateTile(tile, state_dim);
        PreparedTile<value_t> prepared;
        prepared.layout = BuildLayout(tile, state_dim);
        prepared.operations.reserve(tile.gates.size());
        for (const auto& gate : tile.gates) {
            prepared.operations.push_back(
                PrepareOperation(BuildOperation(*gate, tile.qubits, parameter_resolver), prepared.layout.dimension));
        }
        prepared.operations = FoldDiagonalRuns(std::move(prepared.operations), prepared.layout.dimension);
        for (const auto& operation : prepared.operations) {
            prepared.layout.workspace_size = std::max(prepared.layout.workspace_size, WorkspaceSize(operation.kernel));
        }
        return prepared;
    }

    [[nodiscard]] static auto BuildAdjoint(const CircuitTile& tile,
                                           const parameter::ParameterResolver& parameter_resolver,
                                           const MST<size_t>& parameter_map, index_t state_dim)
        -> PreparedAdjointTile<value_t> {
        ValidateTile(tile, state_dim);
        PreparedAdjointTile<value_t> prepared;
        prepared.layout = BuildLayout(tile, state_dim);
        prepared.operations.reserve(tile.gates.size());

        for (const auto& gate : tile.gates) {
            auto evolution = BuildOperation(*gate, tile.qubits, parameter_resolver);
            AdjointOperation<value_t> adjoint;
            if (gate->GradRequired()) {
                auto derivatives = BuildIntrinsicDerivatives(*gate, tile.qubits, parameter_resolver, evolution);
                auto targets = BuildGradientTargets(*gate, parameter_map, derivatives.size());
                for (size_t derivative_index = 0; derivative_index < derivatives.size(); ++derivative_index) {
                    if (targets[derivative_index].empty()) {
                        continue;
                    }
                    auto derivative = PrepareDerivative(std::move(derivatives[derivative_index]),
                                                        prepared.layout.dimension);
                    derivative.targets = std::move(targets[derivative_index]);
                    prepared.layout.workspace_size = std::max(prepared.layout.workspace_size,
                                                              WorkspaceSize(derivative.kernel));
                    adjoint.derivatives.push_back(std::move(derivative));
                }
            }
            adjoint.evolution = PrepareOperation(std::move(evolution), prepared.layout.dimension);
            prepared.layout.workspace_size = std::max(prepared.layout.workspace_size,
                                                      WorkspaceSize(adjoint.evolution.kernel));
            prepared.operations.push_back(std::move(adjoint));
        }
        return prepared;
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
        value_t phase = 1;
        std::array<value_t, 4> matrix2{value_t{1, 0}, value_t{0, 0}, value_t{0, 0}, value_t{1, 0}};
        size_t matrix_dim = 0;
        std::vector<index_t> target_offsets;
        std::vector<value_t> matrix;
    };

    [[nodiscard]] static auto BuildLayout(const CircuitTile& tile, index_t state_dim) -> TileLayout {
        TileLayout layout;
        layout.qubits = tile.qubits;
        layout.dimension = size_t{1} << layout.qubits.size();
        layout.tile_count = state_dim >> layout.qubits.size();
        layout.state_offsets.resize(layout.dimension, 0);
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            for (size_t bit = 0; bit < layout.qubits.size(); ++bit) {
                if ((basis & (size_t{1} << bit)) != 0) {
                    layout.state_offsets[basis] |= index_t{1} << layout.qubits[bit];
                }
            }
        }
        return layout;
    }

    [[nodiscard]] static auto StateQubitCount(index_t dim) -> qbit_t {
        if (dim == 0 || (dim & (dim - 1)) != 0) {
            throw std::invalid_argument("State dimension must be a non-zero power of two.");
        }
        qbit_t count = 0;
        while (dim > 1) {
            dim >>= 1;
            ++count;
        }
        return count;
    }

    static void ValidateTile(const CircuitTile& tile, index_t state_dim) {
        if (tile.gates.empty()) {
            throw std::invalid_argument("A circuit tile cannot be empty.");
        }
        if (tile.qubits.size() >= std::numeric_limits<size_t>::digits) {
            throw std::invalid_argument("Circuit tile is too large for local indexing.");
        }
        const auto state_qubits = StateQubitCount(state_dim);
        if (!std::is_sorted(tile.qubits.begin(), tile.qubits.end())) {
            throw std::invalid_argument("Circuit tile qubits must be sorted.");
        }
        for (size_t index = 0; index < tile.qubits.size(); ++index) {
            const auto qubit = tile.qubits[index];
            if (qubit < 0 || qubit >= state_qubits || (index != 0 && qubit == tile.qubits[index - 1])) {
                throw std::invalid_argument("Circuit tile qubits are invalid for the state.");
            }
        }
        if (std::find(tile.gates.begin(), tile.gates.end(), nullptr) != tile.gates.end()) {
            throw std::invalid_argument("A circuit tile cannot contain a null gate.");
        }
    }

    [[nodiscard]] static auto ResolveScalar(const parameter::ParameterResolver& coefficient,
                                            const parameter::ParameterResolver& parameter_resolver) -> calc_type {
        return tensor::ops::cpu::to_vector<calc_type>(coefficient.Combination(parameter_resolver).const_value)[0];
    }

    static void RequireObjectCount(const BasicGate& gate, size_t expected) {
        const auto actual = gate.GetObjQubits().size();
        if (actual != expected) {
            throw std::invalid_argument(
                fmt::format("Gate {} requires {} object qubits, but got {}.", gate.GetID(), expected, actual));
        }
    }

    [[nodiscard]] static auto LocalMask(const qbits_t& tile_qubits, qbit_t qubit) -> index_t {
        const auto position = std::lower_bound(tile_qubits.begin(), tile_qubits.end(), qubit);
        if (position == tile_qubits.end() || *position != qubit) {
            return 0;
        }
        return index_t{1} << std::distance(tile_qubits.begin(), position);
    }

    static void ConfigureControls(const BasicGate& gate, const qbits_t& tile_qubits, RawOperation* operation) {
        for (auto control : gate.GetCtrlQubits()) {
            AddConditionQubit(control, tile_qubits, operation);
        }
    }

    static void AddConditionQubit(qbit_t qubit, const qbits_t& tile_qubits, RawOperation* operation) {
        const auto local = LocalMask(tile_qubits, qubit);
        if (local != 0) {
            operation->local_control_mask |= local;
        } else {
            operation->external_control_mask |= index_t{1} << qubit;
        }
    }

    static void SetLocalOrExternalZ(qbit_t qubit, const qbits_t& tile_qubits, RawOperation* operation) {
        const auto local = LocalMask(tile_qubits, qubit);
        if (local != 0) {
            operation->pauli_z_mask |= local;
        } else {
            operation->external_pauli_z_mask |= index_t{1} << qubit;
        }
    }

    static void SetLocalPauli(qbit_t qubit, char pauli, const qbits_t& tile_qubits, RawOperation* operation) {
        const auto local = LocalMask(tile_qubits, qubit);
        switch (pauli) {
            case 'X':
                assert(local != 0);
                operation->pauli_x_mask |= local;
                break;
            case 'Y':
                assert(local != 0);
                operation->pauli_y_mask |= local;
                break;
            case 'Z':
                SetLocalOrExternalZ(qubit, tile_qubits, operation);
                break;
            default:
                assert(false && "Unexpected Pauli label");
                std::terminate();
        }
    }

    static void ConfigurePauliMask(const PauliMask& mask, const qbits_t& tile_qubits, RawOperation* operation) {
        for (qbit_t qubit = 0; qubit < static_cast<qbit_t>(std::numeric_limits<index_t>::digits); ++qubit) {
            const auto global = index_t{1} << qubit;
            if ((mask.mask_x & global) != 0) {
                SetLocalPauli(qubit, 'X', tile_qubits, operation);
            } else if ((mask.mask_y & global) != 0) {
                SetLocalPauli(qubit, 'Y', tile_qubits, operation);
            } else if ((mask.mask_z & global) != 0) {
                SetLocalPauli(qubit, 'Z', tile_qubits, operation);
            }
        }
    }

    [[nodiscard]] static auto MatrixFromTensor(const tensor::Matrix& matrix, size_t expected_dim)
        -> std::vector<value_t> {
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

    [[nodiscard]] static auto CustomMatrix(const BasicGate& gate,
                                           const parameter::ParameterResolver& parameter_resolver)
        -> std::vector<value_t> {
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

    [[nodiscard]] static auto MatrixOperation(const BasicGate& gate, const qbits_t& tile_qubits,
                                              std::vector<value_t> matrix) -> RawOperation {
        RawOperation operation;
        ConfigureControls(gate, tile_qubits, &operation);
        const auto objects = gate.GetObjQubits();
        operation.matrix_dim = size_t{1} << objects.size();
        if (matrix.size() != operation.matrix_dim * operation.matrix_dim) {
            throw std::invalid_argument("Matrix operation dimensions are inconsistent with its object qubits.");
        }
        if (objects.size() == 1) {
            operation.kernel = RawKernel::PairMatrix;
            operation.target_mask = LocalMask(tile_qubits, objects.front());
            assert(operation.target_mask != 0);
            std::copy_n(matrix.begin(), 4, operation.matrix2.begin());
            operation.matrix_dim = 0;
            return operation;
        }

        operation.kernel = RawKernel::DenseMatrix;
        operation.matrix = std::move(matrix);
        operation.target_offsets.resize(operation.matrix_dim, 0);
        for (size_t basis = 0; basis < operation.matrix_dim; ++basis) {
            for (size_t bit = 0; bit < objects.size(); ++bit) {
                if ((basis & (size_t{1} << bit)) != 0) {
                    const auto local = LocalMask(tile_qubits, objects[bit]);
                    assert(local != 0);
                    operation.target_offsets[basis] |= local;
                }
            }
        }
        return operation;
    }

    static void ConfigureRotation(const BasicGate& gate, const qbits_t& tile_qubits,
                                  const parameter::ParameterResolver& parameter_resolver, RawOperation* operation) {
        const auto objects = gate.GetObjQubits();
        switch (gate.GetID()) {
            case GateID::RX:
                RequireObjectCount(gate, 1);
                SetLocalPauli(objects[0], 'X', tile_qubits, operation);
                break;
            case GateID::RY:
                RequireObjectCount(gate, 1);
                SetLocalPauli(objects[0], 'Y', tile_qubits, operation);
                break;
            case GateID::RZ:
                RequireObjectCount(gate, 1);
                SetLocalPauli(objects[0], 'Z', tile_qubits, operation);
                break;
            case GateID::Rxx:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'X', tile_qubits, operation);
                SetLocalPauli(objects[1], 'X', tile_qubits, operation);
                break;
            case GateID::Ryy:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'Y', tile_qubits, operation);
                SetLocalPauli(objects[1], 'Y', tile_qubits, operation);
                break;
            case GateID::Rzz:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'Z', tile_qubits, operation);
                SetLocalPauli(objects[1], 'Z', tile_qubits, operation);
                break;
            case GateID::Rxy:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'X', tile_qubits, operation);
                SetLocalPauli(objects[1], 'Y', tile_qubits, operation);
                break;
            case GateID::Rxz:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'X', tile_qubits, operation);
                SetLocalPauli(objects[1], 'Z', tile_qubits, operation);
                break;
            case GateID::Ryz:
                RequireObjectCount(gate, 2);
                SetLocalPauli(objects[0], 'Y', tile_qubits, operation);
                SetLocalPauli(objects[1], 'Z', tile_qubits, operation);
                break;
            case GateID::RPS:
                ConfigurePauliMask(static_cast<const RotPauliString*>(&gate)->GetPauliString().GetPauliMask(),
                                   tile_qubits, operation);
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
                                             const parameter::ParameterResolver& parameter_resolver) -> RawOperation {
        RawOperation operation;
        ConfigureControls(gate, tile_qubits, &operation);
        const auto objects = gate.GetObjQubits();
        constexpr auto half = calc_type{0.5};
        switch (gate.GetID()) {
            case GateID::X:
            case GateID::Y:
            case GateID::Z: {
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::Pauli;
                const auto pauli = gate.GetID() == GateID::X ? 'X' : (gate.GetID() == GateID::Y ? 'Y' : 'Z');
                SetLocalPauli(objects[0], pauli, tile_qubits, &operation);
                break;
            }
            case GateID::PauliString:
                operation.kernel = RawKernel::Pauli;
                ConfigurePauliMask(static_cast<const PauliString*>(&gate)->GetPauliMask(), tile_qubits, &operation);
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
                ConfigureRotation(gate, tile_qubits, parameter_resolver, &operation);
                break;
            case GateID::H:
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::Hadamard;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                assert(operation.target_mask != 0);
                break;
            case GateID::S:
            case GateID::Sdag:
            case GateID::T:
            case GateID::Tdag:
            case GateID::PS: {
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::Phase;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                if (operation.target_mask == 0) {
                    operation.external_control_mask |= index_t{1} << objects[0];
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
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::Phase;
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                operation.phase = std::exp(value_t{0, -angle});
                break;
            }
            case GateID::SX:
            case GateID::SXdag: {
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                assert(operation.target_mask != 0);
                const auto sign = gate.GetID() == GateID::SXdag ? calc_type{-1} : calc_type{1};
                const auto diagonal = value_t{half, sign * half};
                const auto off_diagonal = value_t{half, -sign * half};
                operation.matrix2 = {diagonal, off_diagonal, off_diagonal, diagonal};
                break;
            }
            case GateID::SWAP:
                RequireObjectCount(gate, 2);
                operation.kernel = RawKernel::Swap;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                operation.target_mask_2 = LocalMask(tile_qubits, objects[1]);
                assert(operation.target_mask != 0 && operation.target_mask_2 != 0);
                break;
            case GateID::ISWAP:
                RequireObjectCount(gate, 2);
                operation.kernel = RawKernel::ISwap;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                operation.target_mask_2 = LocalMask(tile_qubits, objects[1]);
                assert(operation.target_mask != 0 && operation.target_mask_2 != 0);
                operation.phase = static_cast<const ISWAPGate*>(&gate)->Daggered() ? value_t{0, -1} : value_t{0, 1};
                break;
            case GateID::SWAPalpha: {
                RequireObjectCount(gate, 2);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                operation.target_mask_2 = LocalMask(tile_qubits, objects[1]);
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto exponential = std::exp(value_t{0, static_cast<calc_type>(M_PI) * angle});
                const auto diagonal = (value_t{1, 0} + exponential) / calc_type{2};
                const auto off_diagonal = (value_t{1, 0} - exponential) / calc_type{2};
                operation.matrix2 = {diagonal, off_diagonal, off_diagonal, diagonal};
                break;
            }
            case GateID::Givens: {
                RequireObjectCount(gate, 2);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                operation.target_mask_2 = LocalMask(tile_qubits, objects[1]);
                const auto angle = ResolveScalar(static_cast<const Parameterizable*>(&gate)->GetCoeffs()[0],
                                                 parameter_resolver);
                const auto cosine = static_cast<calc_type>(std::cos(angle));
                const auto sine = static_cast<calc_type>(std::sin(angle));
                operation.matrix2 = {cosine, -sine, sine, cosine};
                break;
            }
            case GateID::FSim: {
                RequireObjectCount(gate, 2);
                operation.kernel = RawKernel::FSim;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
                operation.target_mask_2 = LocalMask(tile_qubits, objects[1]);
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
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
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
                RequireObjectCount(gate, 1);
                operation.kernel = RawKernel::PairMatrix;
                operation.target_mask = LocalMask(tile_qubits, objects[0]);
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
                return MatrixOperation(gate, tile_qubits, CustomMatrix(gate, parameter_resolver));
            default:
                throw std::invalid_argument(fmt::format("Gate {} cannot be lowered to a circuit tile.", gate.GetID()));
        }
        return operation;
    }

    [[nodiscard]] static auto BuildIntrinsicDerivatives(const BasicGate& gate, const qbits_t& tile_qubits,
                                                        const parameter::ParameterResolver& parameter_resolver,
                                                        const RawOperation& evolution) -> std::vector<RawOperation> {
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
                    MatrixOperation(gate, tile_qubits, MatrixFromTensor(U3DiffThetaMatrix(theta, phi, lambda), 2)),
                    MatrixOperation(gate, tile_qubits, MatrixFromTensor(U3DiffPhiMatrix(theta, phi, lambda), 2)),
                    MatrixOperation(gate, tile_qubits, MatrixFromTensor(U3DiffLambdaMatrix(theta, phi, lambda), 2))};
            }
            case GateID::Rn: {
                const auto* rn = static_cast<const Rn*>(&gate);
                const auto alpha = rn->GetAlpha().Combination(parameter_resolver).const_value;
                const auto beta = rn->GetBeta().Combination(parameter_resolver).const_value;
                const auto gamma = rn->GetGamma().Combination(parameter_resolver).const_value;
                return {MatrixOperation(gate, tile_qubits, MatrixFromTensor(RnDiffAlphaMatrix(alpha, beta, gamma), 2)),
                        MatrixOperation(gate, tile_qubits, MatrixFromTensor(RnDiffBetaMatrix(alpha, beta, gamma), 2)),
                        MatrixOperation(gate, tile_qubits, MatrixFromTensor(RnDiffGammaMatrix(alpha, beta, gamma), 2))};
            }
            case GateID::FSim: {
                RequireObjectCount(gate, 2);
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
                ConfigureControls(gate, tile_qubits, &phi_derivative);
                AddConditionQubit(objects[0], tile_qubits, &phi_derivative);
                AddConditionQubit(objects[1], tile_qubits, &phi_derivative);
                phi_derivative.phase = value_t{0, -1} * std::exp(value_t{0, -phi});
                return {std::move(theta_derivative), std::move(phi_derivative)};
            }
            case GateID::CUSTOM: {
                const auto* custom = static_cast<const CustomGate*>(&gate);
                const auto value = ResolveScalar(custom->GetCoeffs()[0], parameter_resolver);
                const auto expected_dim = size_t{1} << objects.size();
                return {MatrixOperation(gate, tile_qubits,
                                        MatrixFromTensor(custom->GetDiffMatrixWrapper()(value), expected_dim))};
            }
            case GateID::CUSTOM_TWO_PARAM: {
                const auto* custom = static_cast<const CustomTwoParamGate*>(&gate);
                const auto coefficients = custom->GetCoeffs();
                const auto first = ResolveScalar(coefficients[0], parameter_resolver);
                const auto second = ResolveScalar(coefficients[1], parameter_resolver);
                const auto expected_dim = size_t{1} << objects.size();
                return {
                    MatrixOperation(gate, tile_qubits,
                                    MatrixFromTensor(custom->GetDiffMatrix1Wrapper()(first, second), expected_dim)),
                    MatrixOperation(gate, tile_qubits,
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

    [[nodiscard]] static auto LocalPauliPhase(const RawOperation& operation, size_t basis) -> value_t {
        auto phase = value_t{1, 0};
        auto y_mask = operation.pauli_y_mask;
        while (y_mask != 0) {
            const auto bit = y_mask & (~y_mask + 1);
            phase *= (basis & bit) != 0 ? value_t{0, -1} : value_t{0, 1};
            y_mask &= y_mask - 1;
        }
        auto z_mask = operation.pauli_z_mask;
        while (z_mask != 0) {
            const auto bit = z_mask & (~z_mask + 1);
            if ((basis & bit) != 0) {
                phase = -phase;
            }
            z_mask &= z_mask - 1;
        }
        return phase;
    }

    template <typename pauli_kernel_t>
    static void PreparePauliEntries(const RawOperation& raw, size_t tile_dim, pauli_kernel_t* kernel) {
        const auto controls_match = [&raw](size_t basis) {
            return (basis & raw.local_control_mask) == raw.local_control_mask;
        };
        const auto flip_mask = raw.pauli_x_mask | raw.pauli_y_mask;
        if (flip_mask == 0) {
            for (size_t basis = 0; basis < tile_dim; ++basis) {
                if (controls_match(basis)) {
                    kernel->scales.push_back({basis, LocalPauliPhase(raw, basis)});
                }
            }
            return;
        }
        for (size_t lhs = 0; lhs < tile_dim; ++lhs) {
            if (!controls_match(lhs)) {
                continue;
            }
            const auto rhs = lhs ^ flip_mask;
            if (lhs < rhs) {
                kernel->pairs.push_back({{lhs, rhs}, LocalPauliPhase(raw, rhs), LocalPauliPhase(raw, lhs)});
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

    [[nodiscard]] static auto PrepareDense(RawOperation raw, size_t tile_dim) -> DenseMatrixKernel<value_t> {
        assert(raw.matrix_dim != 0 && raw.target_offsets.size() == raw.matrix_dim);
        assert(raw.matrix.size() == raw.matrix_dim * raw.matrix_dim);
        index_t object_mask = 0;
        for (auto offset : raw.target_offsets) {
            object_mask |= offset;
        }

        DenseMatrixKernel<value_t> kernel;
        kernel.matrix_dim = raw.matrix_dim;
        kernel.matrix = std::move(raw.matrix);
        for (size_t basis = 0; basis < tile_dim; ++basis) {
            if ((basis & raw.local_control_mask) != raw.local_control_mask || (basis & object_mask) != 0) {
                continue;
            }
            for (auto offset : raw.target_offsets) {
                kernel.indices.push_back(basis | offset);
            }
        }
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

    [[nodiscard]] static auto FoldDiagonalRuns(std::vector<Operation<value_t>> operations, size_t tile_dim)
        -> std::vector<Operation<value_t>> {
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

    [[nodiscard]] static auto WorkspaceSize(const DerivativeKernel<value_t>& kernel) -> size_t {
        if (const auto* dense = std::get_if<DenseMatrixKernel<value_t>>(&kernel); dense != nullptr) {
            return dense->matrix_dim;
        }
        return 0;
    }
};
}  // namespace mindquantum::sim::vector::detail::circuit_tiling

#endif  // INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_BUILDER_H_
