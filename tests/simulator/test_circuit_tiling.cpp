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

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "simulator/circuit_tiling/builder.h"
#include "simulator/circuit_tiling/planner.h"

#include <catch2/catch_test_macros.hpp>

namespace mq = mindquantum;
namespace mqt = mindquantum::sim::circuit_tiling;

auto VectorSpace(mq::qbit_t n_qubits) -> mqt::StateSpace {
    return mqt::StateSpace::Vector(n_qubits);
}

namespace {
using circuit_t = mqt::Planner::circuit_t;

auto Flatten(const mqt::Schedule& plan) -> circuit_t {
    circuit_t gates;
    for (const auto& operation : plan) {
        if (const auto* tile = std::get_if<mqt::CircuitTile>(&operation); tile != nullptr) {
            gates.insert(gates.end(), tile->gates.begin(), tile->gates.end());
        } else {
            gates.push_back(std::get<std::shared_ptr<mq::BasicGate>>(operation));
        }
    }
    return gates;
}

auto Position(const circuit_t& circuit, const std::shared_ptr<mq::BasicGate>& gate) -> size_t {
    return static_cast<size_t>(std::distance(circuit.begin(), std::find(circuit.begin(), circuit.end(), gate)));
}

void CheckSamePlan(const mqt::Schedule& lhs, const mqt::Schedule& rhs) {
    REQUIRE(lhs.size() == rhs.size());
    for (size_t operation = 0; operation < lhs.size(); ++operation) {
        const auto* lhs_tile = std::get_if<mqt::CircuitTile>(&lhs[operation]);
        const auto* rhs_tile = std::get_if<mqt::CircuitTile>(&rhs[operation]);
        REQUIRE((lhs_tile == nullptr) == (rhs_tile == nullptr));
        if (lhs_tile != nullptr) {
            CHECK(lhs_tile->qubits == rhs_tile->qubits);
            CHECK(lhs_tile->gates == rhs_tile->gates);
        } else {
            CHECK(std::get<std::shared_ptr<mq::BasicGate>>(lhs[operation])
                  == std::get<std::shared_ptr<mq::BasicGate>>(rhs[operation]));
        }
    }
}
}  // namespace

TEST_CASE("Circuit tiles use the minimum passes for independent repeated gates") {
    circuit_t circuit{std::make_shared<mq::HGate>(mq::qbits_t{0}), std::make_shared<mq::HGate>(mq::qbits_t{1}),
                      std::make_shared<mq::HGate>(mq::qbits_t{2}), std::make_shared<mq::HGate>(mq::qbits_t{0}),
                      std::make_shared<mq::HGate>(mq::qbits_t{1}), std::make_shared<mq::HGate>(mq::qbits_t{2})};

    const auto plan = mqt::Planner(VectorSpace(12), 2).Plan(circuit);

    REQUIRE(plan.size() == 2);
    for (const auto& operation : plan) {
        REQUIRE(std::holds_alternative<mqt::CircuitTile>(operation));
        const auto& tile = std::get<mqt::CircuitTile>(operation);
        CHECK(tile.qubits.size() <= 2);
        CHECK((tile.gates.size() == 2 || tile.gates.size() == 4));
    }
}

TEST_CASE("Diagonal gates share a pass without consuming mixing capacity") {
    circuit_t circuit{std::make_shared<mq::ZGate>(mq::qbits_t{0}), std::make_shared<mq::SGate>(mq::qbits_t{7}),
                      std::make_shared<mq::PauliString>("IZ", mq::qbits_t{5, 9})};

    const auto plan = mqt::Planner(VectorSpace(12), 1).Plan(circuit);

    REQUIRE(plan.size() == 1);
    REQUIRE(std::holds_alternative<mqt::CircuitTile>(plan.front()));
    const auto& tile = std::get<mqt::CircuitTile>(plan.front());
    CHECK(tile.qubits.size() == 1);
    CHECK(tile.gates.size() == circuit.size());
}

TEST_CASE("A diagonal frontier can expand through the full block width") {
    circuit_t circuit;
    for (mq::qbit_t qubit = 0; qubit < 4; ++qubit) {
        circuit.push_back(std::make_shared<mq::ZGate>(mq::qbits_t{qubit}));
    }
    for (mq::qbit_t qubit = 0; qubit < 4; ++qubit) {
        circuit.push_back(std::make_shared<mq::HGate>(mq::qbits_t{qubit}));
    }

    const auto plan = mqt::Planner(VectorSpace(12), 4).Plan(circuit);

    REQUIRE(plan.size() == 1);
    REQUIRE(std::holds_alternative<mqt::CircuitTile>(plan.front()));
    const auto& tile = std::get<mqt::CircuitTile>(plan.front());
    CHECK(tile.qubits.size() == 4);
    CHECK(tile.gates.size() == circuit.size());
}

TEST_CASE("The default circuit tile holds eight mixing qubits") {
    circuit_t circuit;
    for (mq::qbit_t qubit = 0; qubit < 8; ++qubit) {
        circuit.push_back(std::make_shared<mq::HGate>(mq::qbits_t{qubit}));
    }

    const auto plan = mqt::Planner(VectorSpace(13)).Plan(circuit);

    REQUIRE(plan.size() == 1);
    REQUIRE(std::holds_alternative<mqt::CircuitTile>(plan.front()));
    const auto& tile = std::get<mqt::CircuitTile>(plan.front());
    CHECK(tile.qubits.size() == 8);
    CHECK(tile.gates.size() == circuit.size());
}

TEST_CASE("Bounded closure search is efficient and deterministic") {
    circuit_t circuit;
    for (mq::qbit_t layer = 0; layer < 8; ++layer) {
        for (mq::qbit_t offset = 0; offset < 8; ++offset) {
            const auto target = static_cast<mq::qbit_t>((offset + layer) % 8);
            circuit.push_back(std::make_shared<mq::HGate>(mq::qbits_t{target}));
            if (offset % 2 == 0) {
                circuit.push_back(std::make_shared<mq::ZGate>(mq::qbits_t{static_cast<mq::qbit_t>((target + 3) % 8)}));
            }
            if (offset % 3 == 0) {
                circuit.push_back(std::make_shared<mq::XGate>(mq::qbits_t{target},
                                                              mq::qbits_t{static_cast<mq::qbit_t>((target + 4) % 8)}));
            }
        }
    }

    const mqt::Planner planner(VectorSpace(12), 4);
    const auto expected = planner.Plan(circuit);
    REQUIRE(expected.size() == 2);
    const auto flattened = Flatten(expected);
    CHECK(flattened.size() == circuit.size());
    CHECK(std::is_permutation(flattened.begin(), flattened.end(), circuit.begin(), circuit.end()));
    for (size_t repeat = 0; repeat < 3; ++repeat) {
        CheckSamePlan(expected, planner.Plan(circuit));
    }
}

TEST_CASE("Bounded closure search preserves future tile choices") {
    enum class Kind : uint8_t { X, ControlledX, Z, ControlledZ, H, Swap };
    struct GateSpec {
        Kind kind;
        mq::qbit_t first;
        mq::qbit_t second = -1;
    };
    constexpr std::array specs{
        GateSpec{Kind::X, 4},
        GateSpec{Kind::ControlledX, 0, 4},
        GateSpec{Kind::Z, 3},
        GateSpec{Kind::X, 0},
        GateSpec{Kind::H, 3},
        GateSpec{Kind::ControlledZ, 0, 4},
        GateSpec{Kind::ControlledX, 5, 1},
        GateSpec{Kind::ControlledZ, 0, 3},
        GateSpec{Kind::H, 0},
        GateSpec{Kind::Swap, 4, 0},
        GateSpec{Kind::Z, 5},
        GateSpec{Kind::Z, 5},
        GateSpec{Kind::ControlledZ, 1, 4},
        GateSpec{Kind::Z, 4},
        GateSpec{Kind::ControlledX, 1, 2},
        GateSpec{Kind::Z, 2},
        GateSpec{Kind::Z, 4},
        GateSpec{Kind::X, 5},
        GateSpec{Kind::H, 5},
        GateSpec{Kind::Swap, 5, 4},
        GateSpec{Kind::Z, 4},
        GateSpec{Kind::ControlledX, 2, 5},
        GateSpec{Kind::Z, 4},
        GateSpec{Kind::ControlledZ, 0, 4},
        GateSpec{Kind::X, 5},
        GateSpec{Kind::Z, 5},
        GateSpec{Kind::ControlledX, 4, 2},
        GateSpec{Kind::H, 3},
        GateSpec{Kind::H, 1},
        GateSpec{Kind::Z, 2},
        GateSpec{Kind::Swap, 0, 4},
        GateSpec{Kind::H, 2},
        GateSpec{Kind::ControlledZ, 4, 3},
        GateSpec{Kind::Swap, 1, 2},
        GateSpec{Kind::ControlledZ, 1, 0},
        GateSpec{Kind::X, 4},
    };

    circuit_t circuit;
    circuit.reserve(specs.size());
    for (const auto& spec : specs) {
        switch (spec.kind) {
            case Kind::X:
                circuit.push_back(std::make_shared<mq::XGate>(mq::qbits_t{spec.first}));
                break;
            case Kind::ControlledX:
                circuit.push_back(std::make_shared<mq::XGate>(mq::qbits_t{spec.first}, mq::qbits_t{spec.second}));
                break;
            case Kind::Z:
                circuit.push_back(std::make_shared<mq::ZGate>(mq::qbits_t{spec.first}));
                break;
            case Kind::ControlledZ:
                circuit.push_back(std::make_shared<mq::ZGate>(mq::qbits_t{spec.first}, mq::qbits_t{spec.second}));
                break;
            case Kind::H:
                circuit.push_back(std::make_shared<mq::HGate>(mq::qbits_t{spec.first}));
                break;
            case Kind::Swap:
                circuit.push_back(std::make_shared<mq::SWAPGate>(mq::qbits_t{spec.first, spec.second}));
                break;
        }
    }

    const auto plan = mqt::Planner(VectorSpace(6), 3).Plan(circuit);

    REQUIRE(plan.size() == 4);
    const auto flattened = Flatten(plan);
    CHECK(std::is_permutation(flattened.begin(), flattened.end(), circuit.begin(), circuit.end()));
}

TEST_CASE("Mixing dependencies preserve diagonal and controlled operations") {
    auto first_target_mix = std::make_shared<mq::HGate>(mq::qbits_t{0});
    auto diagonal = std::make_shared<mq::ZGate>(mq::qbits_t{0});
    auto second_target_mix = std::make_shared<mq::HGate>(mq::qbits_t{0});
    auto first_control_mix = std::make_shared<mq::HGate>(mq::qbits_t{8});
    auto controlled = std::make_shared<mq::XGate>(mq::qbits_t{1}, mq::qbits_t{8});
    auto second_control_mix = std::make_shared<mq::HGate>(mq::qbits_t{8});
    circuit_t circuit{first_target_mix,
                      diagonal,
                      second_target_mix,
                      first_control_mix,
                      controlled,
                      second_control_mix,
                      std::make_shared<mq::HGate>(mq::qbits_t{11})};

    const auto scheduled = Flatten(mqt::Planner(VectorSpace(12), 2).Plan(circuit));

    CHECK(Position(scheduled, first_target_mix) < Position(scheduled, diagonal));
    CHECK(Position(scheduled, diagonal) < Position(scheduled, second_target_mix));
    CHECK(Position(scheduled, first_control_mix) < Position(scheduled, controlled));
    CHECK(Position(scheduled, controlled) < Position(scheduled, second_control_mix));
}

TEST_CASE("Non-unitary gates delimit independently scheduled segments") {
    auto measurement = std::make_shared<mq::MeasureGate>("m", mq::qbits_t{0});
    circuit_t circuit{std::make_shared<mq::HGate>(mq::qbits_t{0}), std::make_shared<mq::HGate>(mq::qbits_t{1}),
                      measurement, std::make_shared<mq::XGate>(mq::qbits_t{0}),
                      std::make_shared<mq::XGate>(mq::qbits_t{1})};

    const auto plan = mqt::Planner(VectorSpace(12), 2).Plan(circuit);

    REQUIRE(plan.size() == 3);
    CHECK(std::holds_alternative<mqt::CircuitTile>(plan[0]));
    CHECK(std::get<std::shared_ptr<mq::BasicGate>>(plan[1]) == measurement);
    CHECK(std::holds_alternative<mqt::CircuitTile>(plan[2]));
}

TEST_CASE("Identity gates do not create state passes") {
    circuit_t circuit{std::make_shared<mq::IGate>(mq::qbits_t{0}), std::make_shared<mq::HGate>(mq::qbits_t{1}),
                      std::make_shared<mq::IGate>(mq::qbits_t{2})};

    const auto plan = mqt::Planner(VectorSpace(12), 2).Plan(circuit);

    REQUIRE(plan.size() == 1);
    CHECK(std::get<std::shared_ptr<mq::BasicGate>>(plan.front())->GetID() == mq::GateID::H);
}

TEST_CASE("Circuit tile builder folds local diagonal runs") {
    mqt::CircuitTile tile;
    tile.qubits = {0, 7};
    tile.gates = {std::make_shared<mq::ZGate>(mq::qbits_t{0}), std::make_shared<mq::SGate>(mq::qbits_t{7}),
                  std::make_shared<mq::TGate>(mq::qbits_t{0})};

    const auto prepared = mqt::Builder<std::complex<double>>::Build(tile, parameter::ParameterResolver{},
                                                                    VectorSpace(12));

    REQUIRE(prepared.operations.size() == 1);
    using diagonal_t = mqt::DiagonalKernel<std::complex<double>>;
    REQUIRE(std::holds_alternative<diagonal_t>(prepared.operations.front().kernel));
    const auto& diagonal = std::get<diagonal_t>(prepared.operations.front().kernel);
    CHECK_FALSE(diagonal.scales.empty());
}

TEST_CASE("Circuit tile builder prepares local Pauli rotation coefficients") {
    mqt::CircuitTile tile;
    tile.qubits = {3};
    tile.gates = {std::make_shared<mq::RXGate>(parameter::ParameterResolver{0.37}, mq::qbits_t{3})};

    const auto prepared = mqt::Builder<std::complex<double>>::Build(tile, parameter::ParameterResolver{},
                                                                    VectorSpace(12));

    REQUIRE(prepared.operations.size() == 1);
    using rotation_t = mqt::PauliRotationKernel<std::complex<double>>;
    REQUIRE(std::holds_alternative<rotation_t>(prepared.operations.front().kernel));
    const auto& rotation = std::get<rotation_t>(prepared.operations.front().kernel);
    CHECK(rotation.coefficients_prepared);
}

TEST_CASE("Identity Pauli strings lower to an empty circuit tile") {
    mqt::CircuitTile tile;
    tile.gates = {std::make_shared<mq::PauliString>("III", mq::qbits_t{0, 4, 9}),
                  std::make_shared<mq::PauliString>("II", mq::qbits_t{2, 7})};

    const auto prepared = mqt::Builder<std::complex<double>>::Build(tile, parameter::ParameterResolver{},
                                                                    VectorSpace(12));

    CHECK(prepared.operations.empty());
}
