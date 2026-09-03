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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_STATE_SPACE_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_STATE_SPACE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"

namespace mindquantum::sim::circuit_tiling {
// One placement of the circuit qubits inside the state index.  A unitary U
// acts on every image of its qubits; on a conjugated image it acts as
// conj(U), which is how a row-major density matrix realises the right-hand
// factor of U rho U^dagger.
struct QubitImage {
    qbit_t offset = 0;
    bool conjugated = false;
};

// Describes how circuit qubits index the simulated state.
//
//  - A state vector embeds circuit qubit q as state-index bit q.
//  - A row-major density matrix rho[i][j] = state[i * dim + j] embeds circuit
//    qubit q twice: as column-index bit q and as row-index bit q + n.
class StateSpace {
 public:
    enum class Kind : std::uint8_t { Vector, DensityMatrix };

    [[nodiscard]] static auto Vector(qbit_t n_qubits) -> StateSpace {
        return StateSpace(Kind::Vector, n_qubits);
    }

    [[nodiscard]] static auto DensityMatrix(qbit_t n_qubits) -> StateSpace {
        return StateSpace(Kind::DensityMatrix, n_qubits);
    }

    [[nodiscard]] auto kind() const noexcept -> Kind {
        return kind_;
    }

    [[nodiscard]] auto circuit_qubits() const noexcept -> qbit_t {
        return circuit_qubits_;
    }

    [[nodiscard]] auto state_qubits() const noexcept -> qbit_t {
        return kind_ == Kind::Vector ? circuit_qubits_ : 2 * circuit_qubits_;
    }

    [[nodiscard]] auto dimension() const noexcept -> index_t {
        return index_t{1} << state_qubits();
    }

    // Images in application order.  The last image is never conjugated and
    // is the one that carries a gate's derivative in adjoint differentiation.
    [[nodiscard]] auto Images() const -> std::vector<QubitImage> {
        if (kind_ == Kind::Vector) {
            return {QubitImage{0, false}};
        }
        return {ColumnImage(), RowImage()};
    }

    [[nodiscard]] auto ColumnImage() const noexcept -> QubitImage {
        return QubitImage{0, kind_ == Kind::DensityMatrix};
    }

    [[nodiscard]] auto RowImage() const noexcept -> QubitImage {
        return QubitImage{kind_ == Kind::Vector ? 0 : circuit_qubits_, false};
    }

    // Every state-index bit touched by a circuit-qubit mask.
    [[nodiscard]] auto Embed(index_t circuit_mask) const noexcept -> index_t {
        return kind_ == Kind::Vector ? circuit_mask : (circuit_mask | (circuit_mask << circuit_qubits_));
    }

    [[nodiscard]] auto operator==(const StateSpace& other) const noexcept -> bool {
        return kind_ == other.kind_ && circuit_qubits_ == other.circuit_qubits_;
    }

    [[nodiscard]] auto operator!=(const StateSpace& other) const noexcept -> bool {
        return !(*this == other);
    }

 private:
    StateSpace(Kind kind, qbit_t circuit_qubits) : kind_(kind), circuit_qubits_(circuit_qubits) {
        if (circuit_qubits_ < 0 || state_qubits() >= std::numeric_limits<index_t>::digits) {
            throw std::invalid_argument("Simulator qubit count is outside the state index range.");
        }
    }

    Kind kind_;
    qbit_t circuit_qubits_;
};

[[nodiscard]] inline auto MaskFromQubits(const qbits_t& qubits) noexcept -> index_t {
    index_t mask = 0;
    for (auto qubit : qubits) {
        mask |= index_t{1} << qubit;
    }
    return mask;
}

[[nodiscard]] inline auto QubitsFromMask(index_t mask) -> qbits_t {
    qbits_t qubits;
    qubits.reserve(CountOne(static_cast<std::uint64_t>(mask)));
    while (mask != 0) {
        const auto bit = mask & (~mask + 1);
        qubits.push_back(static_cast<qbit_t>(CountOne(static_cast<std::uint64_t>(bit - 1))));
        mask &= mask - 1;
    }
    return qubits;
}

[[nodiscard]] constexpr auto IsNoiseChannel(GateID gate_id) noexcept -> bool {
    switch (gate_id) {
        case GateID::PL:
        case GateID::GPL:
        case GateID::DEP:
        case GateID::AD:
        case GateID::PD:
        case GateID::KRAUS:
        case GateID::TR:
            return true;
        default:
            return false;
    }
}

// How a gate reads and writes the state, expressed in state-index bits.
struct GateAccess {
    std::shared_ptr<BasicGate> gate;
    index_t mixing_mask = 0;   // bits whose amplitudes the gate mixes
    index_t object_mask = 0;   // bits the gate reads, including diagonal operands
    index_t control_mask = 0;  // bits that condition the gate
};

namespace detail {
enum class AccessPattern : std::uint8_t {
    MixAllObjects,
    DiagonalObjects,
    MixFirstObject,
    PauliString,
    GlobalPhase,
};

[[nodiscard]] inline auto PatternFor(GateID gate_id, StateSpace::Kind kind) -> std::optional<AccessPattern> {
    switch (gate_id) {
        case GateID::X:
        case GateID::Y:
        case GateID::H:
        case GateID::SX:
        case GateID::SXdag:
        case GateID::RX:
        case GateID::RY:
        case GateID::SWAP:
        case GateID::ISWAP:
        case GateID::SWAPalpha:
        case GateID::Rxx:
        case GateID::Ryy:
        case GateID::Rxy:
        case GateID::Givens:
        case GateID::U3:
        case GateID::Rn:
        case GateID::FSim:
        case GateID::CUSTOM:
        case GateID::CUSTOM_TWO_PARAM:
            return AccessPattern::MixAllObjects;
        case GateID::Z:
        case GateID::S:
        case GateID::Sdag:
        case GateID::T:
        case GateID::Tdag:
        case GateID::PS:
        case GateID::RZ:
        case GateID::Rzz:
            return AccessPattern::DiagonalObjects;
        case GateID::Rxz:
        case GateID::Ryz:
            return AccessPattern::MixFirstObject;
        case GateID::PauliString:
        case GateID::RPS:
            return AccessPattern::PauliString;
        case GateID::GP:
            return AccessPattern::GlobalPhase;
        default:
            break;
    }
    // A noise channel is a stochastic gate on a state vector but a linear
    // superoperator on a density matrix, where it mixes every image of its
    // qubits.  Grouped Pauli channels are expanded before lowering.
    if (kind == StateSpace::Kind::DensityMatrix && IsNoiseChannel(gate_id) && gate_id != GateID::GPL) {
        return AccessPattern::MixAllObjects;
    }
    return std::nullopt;
}

inline void ValidateGateQubits(const BasicGate& gate, const qbits_t& objects, const qbits_t& controls,
                               qbit_t n_qubits) {
    const auto validate_set = [&gate, n_qubits](const qbits_t& qubits, const char* kind) {
        auto sorted = qubits;
        for (auto qubit : sorted) {
            if (qubit < 0 || qubit >= n_qubits) {
                throw std::invalid_argument(
                    fmt::format("{} qubit {} of gate {} is out of range.", kind, qubit, gate.GetID()));
            }
        }
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument(fmt::format("{} qubits of gate {} contain duplicates.", kind, gate.GetID()));
        }
    };
    validate_set(objects, "Object");
    validate_set(controls, "Control");

    auto all = objects;
    all.insert(all.end(), controls.begin(), controls.end());
    std::sort(all.begin(), all.end());
    if (std::adjacent_find(all.begin(), all.end()) != all.end()) {
        throw std::invalid_argument(
            fmt::format("Object and control qubits of gate {} must be disjoint.", gate.GetID()));
    }
}
}  // namespace detail

// Describes a gate's state access, or nothing when the gate cannot be
// lowered into a circuit tile of the given space (identity, measurement, and
// stochastic channels on a state vector).
[[nodiscard]] inline auto DescribeGateAccess(const std::shared_ptr<BasicGate>& gate, const StateSpace& space)
    -> std::optional<GateAccess> {
    if (gate == nullptr) {
        throw std::invalid_argument("Cannot describe a null gate.");
    }
    const auto pattern = detail::PatternFor(gate->GetID(), space.kind());
    if (!pattern.has_value()) {
        return std::nullopt;
    }

    const auto objects = gate->GetObjQubits();
    const auto controls = gate->GetCtrlQubits();
    detail::ValidateGateQubits(*gate, objects, controls, space.circuit_qubits());
    auto object_mask = MaskFromQubits(objects);
    index_t mixing_mask = 0;
    switch (*pattern) {
        case detail::AccessPattern::MixAllObjects:
            mixing_mask = object_mask;
            break;
        case detail::AccessPattern::DiagonalObjects:
            break;
        case detail::AccessPattern::MixFirstObject:
            mixing_mask = objects.empty() ? 0 : index_t{1} << objects.front();
            break;
        case detail::AccessPattern::PauliString: {
            const auto mask = gate->GetID() == GateID::PauliString
                                  ? static_cast<const PauliString*>(gate.get())->GetPauliMask()
                                  : static_cast<const RotPauliString*>(gate.get())->GetPauliString().GetPauliMask();
            mixing_mask = mask.mask_x | mask.mask_y;
            // Identity labels in a Pauli string are not state accesses.
            object_mask = mixing_mask | mask.mask_z;
            break;
        }
        case detail::AccessPattern::GlobalPhase:
            // GP's public object label does not affect phase * I.
            object_mask = 0;
            break;
    }
    return GateAccess{gate, space.Embed(mixing_mask), space.Embed(object_mask), space.Embed(MaskFromQubits(controls))};
}
}  // namespace mindquantum::sim::circuit_tiling

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_STATE_SPACE_H_
