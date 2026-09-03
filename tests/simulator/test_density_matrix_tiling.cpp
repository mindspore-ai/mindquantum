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

#include <cmath>

#include <algorithm>
#include <array>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "ops/gates.h"
#include "simulator/circuit_tiling/builder.h"
#include "simulator/circuit_tiling/detail/scalar_runtime.h"
#include "simulator/circuit_tiling/detail/tile_engine.h"
#include "simulator/circuit_tiling/planner.h"
#include "simulator/circuit_tiling/state_space.h"

#include <catch2/catch_test_macros.hpp>

namespace mq = mindquantum;
namespace mqt = mindquantum::sim::circuit_tiling;

namespace {
using value_t = std::complex<double>;
using builder_t = mqt::Builder<value_t>;
using runtime_t = mqt::detail::ScalarTileRuntime<value_t>;
using gate_t = std::shared_ptr<mq::BasicGate>;

constexpr mq::qbit_t kQubits = 3;
constexpr size_t kDim = size_t{1} << kQubits;

auto RandomDensityMatrix() -> std::vector<value_t> {
    // A fixed positive matrix of unit trace; only Hermiticity and trace matter here.
    std::vector<value_t> raw(kDim * kDim);
    unsigned seed = 12345;
    for (auto& entry : raw) {
        seed = seed * 1103515245U + 12345U;
        const auto real = static_cast<double>((seed >> 8) % 1000) / 1000.0 - 0.5;
        seed = seed * 1103515245U + 12345U;
        const auto imag = static_cast<double>((seed >> 8) % 1000) / 1000.0 - 0.5;
        entry = value_t{real, imag};
    }
    std::vector<value_t> rho(kDim * kDim, 0);
    value_t trace = 0;
    for (size_t row = 0; row < kDim; ++row) {
        for (size_t column = 0; column < kDim; ++column) {
            value_t sum = 0;
            for (size_t k = 0; k < kDim; ++k) {
                sum += raw[row * kDim + k] * std::conj(raw[column * kDim + k]);
            }
            rho[row * kDim + column] = sum;
        }
        trace += rho[row * kDim + row];
    }
    for (auto& entry : rho) {
        entry /= trace;
    }
    return rho;
}

// The dense matrix of a gate, obtained by lowering it into the state vector
// space and applying it to every basis vector.
auto GateMatrix(const gate_t& gate, const parameter::ParameterResolver& pr) -> std::vector<value_t> {
    const auto space = mqt::StateSpace::Vector(kQubits);
    const auto access = mqt::DescribeGateAccess(gate, space);
    REQUIRE(access.has_value());
    const mqt::CircuitTile tile{mqt::QubitsFromMask(access->mixing_mask | access->object_mask | access->control_mask),
                                {gate}};
    const auto prepared = builder_t::Build(tile, pr, space);
    std::vector<value_t> matrix(kDim * kDim, 0);
    for (size_t column = 0; column < kDim; ++column) {
        std::vector<value_t> state(kDim, 0);
        state[column] = 1;
        if (!prepared.operations.empty()) {
            mqt::detail::ApplyTile<runtime_t>(state.data(), prepared, false);
        }
        for (size_t row = 0; row < kDim; ++row) {
            matrix[row * kDim + column] = state[row];
        }
    }
    return matrix;
}

auto Sandwich(const std::vector<value_t>& left, const std::vector<value_t>& rho,
              const std::vector<value_t>& right_dagger) -> std::vector<value_t> {
    std::vector<value_t> product(kDim * kDim, 0);
    for (size_t row = 0; row < kDim; ++row) {
        for (size_t column = 0; column < kDim; ++column) {
            value_t sum = 0;
            for (size_t k = 0; k < kDim; ++k) {
                for (size_t l = 0; l < kDim; ++l) {
                    sum += left[row * kDim + k] * rho[k * kDim + l] * std::conj(right_dagger[column * kDim + l]);
                }
            }
            product[row * kDim + column] = sum;
        }
    }
    return product;
}

void CheckClose(const std::vector<value_t>& actual, const std::vector<value_t>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        CAPTURE(index, actual[index], expected[index]);
        CHECK(std::abs(actual[index] - expected[index]) < 1e-12);
    }
}

auto ApplyOnDensityMatrix(const gate_t& gate, const parameter::ParameterResolver& pr, std::vector<value_t> rho,
                          bool minimal_tile) -> std::vector<value_t> {
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const auto access = mqt::DescribeGateAccess(gate, space);
    REQUIRE(access.has_value());
    const auto mask = minimal_tile ? access->mixing_mask
                                   : access->mixing_mask | access->object_mask | access->control_mask;
    const mqt::CircuitTile tile{mqt::QubitsFromMask(mask), {gate}};
    const auto prepared = builder_t::Build(tile, pr, space);
    if (!prepared.operations.empty()) {
        mqt::detail::ApplyTile<runtime_t>(rho.data(), prepared, false);
    }
    return rho;
}
}  // namespace

TEST_CASE("A density matrix space embeds every circuit qubit as a column and a row index bit") {
    const auto space = mqt::StateSpace::DensityMatrix(3);
    CHECK(space.state_qubits() == 6);
    CHECK(space.dimension() == 64);
    CHECK(space.Embed(0b101) == 0b101101);
    const auto images = space.Images();
    REQUIRE(images.size() == 2);
    CHECK(images.front().offset == 0);
    CHECK(images.front().conjugated);
    CHECK(images.back().offset == 3);
    CHECK_FALSE(images.back().conjugated);

    const auto vector = mqt::StateSpace::Vector(3);
    CHECK(vector.state_qubits() == 3);
    CHECK(vector.Embed(0b101) == 0b101);
    CHECK(vector.Images().size() == 1);
}

TEST_CASE("Noise channels are tileable superoperators only on a density matrix") {
    const auto channel = std::make_shared<mq::AmplitudeDampingChannel>(false, 0.3, mq::qbits_t{1});
    CHECK_FALSE(mqt::DescribeGateAccess(channel, mqt::StateSpace::Vector(3)).has_value());
    const auto access = mqt::DescribeGateAccess(channel, mqt::StateSpace::DensityMatrix(3));
    REQUIRE(access.has_value());
    CHECK(access->mixing_mask == 0b010010);
    CHECK(access->object_mask == 0b010010);
    CHECK(access->control_mask == 0);
}

TEST_CASE("Unitary gates evolve a density matrix as U rho U^dagger") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const std::vector<gate_t> gates{
        std::make_shared<mq::HGate>(mq::qbits_t{0}),
        std::make_shared<mq::YGate>(mq::qbits_t{2}, mq::qbits_t{0}),
        std::make_shared<mq::XGate>(mq::qbits_t{1}, mq::qbits_t{2}),
        std::make_shared<mq::RXGate>(parameter::ParameterResolver{0.37}, mq::qbits_t{1}, mq::qbits_t{0}),
        std::make_shared<mq::RYGate>(parameter::ParameterResolver{-1.2}, mq::qbits_t{2}),
        std::make_shared<mq::RzzGate>(parameter::ParameterResolver{0.81}, mq::qbits_t{0, 2}),
        std::make_shared<mq::ISWAPGate>(false, mq::qbits_t{0, 2}),
        std::make_shared<mq::SGate>(mq::qbits_t{1}),
        std::make_shared<mq::RotPauliString>("XYZ", parameter::ParameterResolver{0.6}, mq::qbits_t{0, 1, 2}),
        std::make_shared<mq::PauliString>("YZ", mq::qbits_t{2, 0}, mq::qbits_t{1}),
        std::make_shared<mq::FSim>(parameter::ParameterResolver{0.4}, parameter::ParameterResolver{-0.9},
                                   mq::qbits_t{1, 2}, mq::qbits_t{}),
    };
    for (const auto& gate : gates) {
        const auto unitary = GateMatrix(gate, pr);
        const auto expected = Sandwich(unitary, rho, unitary);
        CheckClose(ApplyOnDensityMatrix(gate, pr, rho, false), expected);
        // Diagonal operands and controls outside the tile take the external mask paths.
        CheckClose(ApplyOnDensityMatrix(gate, pr, rho, true), expected);
    }
}

TEST_CASE("Kraus channels evolve a density matrix as the sum over Kraus operators") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const auto gamma = 0.35;
    const auto channel = std::make_shared<mq::AmplitudeDampingChannel>(true, gamma, mq::qbits_t{1});

    std::vector<value_t> k0(kDim * kDim, 0);
    std::vector<value_t> k1(kDim * kDim, 0);
    for (size_t basis = 0; basis < kDim; ++basis) {
        const bool excited = (basis & 0b010) != 0;
        k0[basis * kDim + basis] = excited ? std::sqrt(1 - gamma) : 1.0;
        if (!excited) {
            // Daggered amplitude damping raises |0> to |1>.
            k1[(basis | 0b010) * kDim + basis] = std::sqrt(gamma);
        }
    }
    auto expected = Sandwich(k0, rho, k0);
    const auto second = Sandwich(k1, rho, k1);
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] += second[index];
    }
    CheckClose(ApplyOnDensityMatrix(channel, pr, rho, false), expected);
}

TEST_CASE("The depolarizing kernel mixes the operand block towards the identity") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const auto probability = 0.2;
    const auto channel = std::make_shared<mq::DepolarizingChannel>(probability, mq::qbits_t{0, 2});

    // (1 - p) rho + p / 4 Tr_{0,2}(rho) (x) I_{0,2}: the identity factor is
    // diagonal on the channel qubits only, so the partial trace over the
    // channel qubits still couples different states of the spectator qubit.
    constexpr size_t kChannelMask = 0b101;
    std::vector<value_t> expected(kDim * kDim, 0);
    for (size_t row = 0; row < kDim; ++row) {
        for (size_t column = 0; column < kDim; ++column) {
            expected[row * kDim + column] = (1 - probability) * rho[row * kDim + column];
            if ((row & kChannelMask) != (column & kChannelMask)) {
                continue;
            }
            value_t partial_trace = 0;
            for (size_t local = 0; local < 4; ++local) {
                const auto placed = (local & 1) | ((local & 2) << 1);
                partial_trace += rho[((row & ~kChannelMask) | placed) * kDim + ((column & ~kChannelMask) | placed)];
            }
            expected[row * kDim + column] += probability / 4.0 * partial_trace;
        }
    }
    CheckClose(ApplyOnDensityMatrix(channel, pr, rho, false), expected);
}

TEST_CASE("Derivative tiles realise dU rho dU^dagger, which vanishes outside the support of dU") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const auto theta = 0.7;
    const auto derivative_of = [&](const gate_t& gate) {
        const auto access = mqt::DescribeGateAccess(gate, space);
        REQUIRE(access.has_value());
        const mqt::CircuitTile tile{
            mqt::QubitsFromMask(access->mixing_mask | access->object_mask | access->control_mask), {gate}};
        auto state = rho;
        mqt::detail::ApplyTile<runtime_t>(state.data(), builder_t::BuildDerivative(tile, pr, space), false);
        return state;
    };
    const auto embed_single = [](const std::array<value_t, 4>& local, size_t qubit, size_t control) {
        std::vector<value_t> matrix(kDim * kDim, 0);
        const auto bit = size_t{1} << qubit;
        for (size_t basis = 0; basis < kDim; ++basis) {
            if (control != kDim && (basis & (size_t{1} << control)) == 0) {
                continue;
            }
            const auto low = basis & ~bit;
            const auto high = basis | bit;
            const bool set = (basis & bit) != 0;
            matrix[low * kDim + basis] += local[set ? 1 : 0];
            matrix[high * kDim + basis] += local[set ? 3 : 2];
        }
        return matrix;
    };

    SECTION("a phase shift derivative is supported on |1> only") {
        const auto gate = std::make_shared<mq::PSGate>(parameter::ParameterResolver{theta}, mq::qbits_t{1});
        const auto d_phase = value_t{0, 1} * std::exp(value_t{0, theta});
        const auto matrix = embed_single({0, 0, 0, d_phase}, 1, kDim);
        CheckClose(derivative_of(gate), Sandwich(matrix, rho, matrix));
    }
    SECTION("a controlled rotation derivative vanishes off the control subspace") {
        const auto gate = std::make_shared<mq::RYGate>(parameter::ParameterResolver{theta}, mq::qbits_t{0},
                                                       mq::qbits_t{2});
        const auto cosine = std::cos(theta / 2) / 2;
        const auto sine = std::sin(theta / 2) / 2;
        // d/dtheta RY(theta) = (1/2) [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
        const auto matrix = embed_single({-sine, -cosine, cosine, -sine}, 0, 2);
        CheckClose(derivative_of(gate), Sandwich(matrix, rho, matrix));
    }
    SECTION("an exchange derivative vanishes on |00> and |11>") {
        const auto gate = std::make_shared<mq::SWAPalphaGate>(parameter::ParameterResolver{theta}, mq::qbits_t{0, 2});
        const auto diagonal = value_t{0, M_PI_2} * std::exp(value_t{0, M_PI * theta});
        std::vector<value_t> matrix(kDim * kDim, 0);
        for (size_t basis = 0; basis < kDim; ++basis) {
            const bool first = (basis & 0b001) != 0;
            const bool second = (basis & 0b100) != 0;
            if (first == second) {
                continue;
            }
            matrix[basis * kDim + basis] += diagonal;
            matrix[(basis ^ 0b101) * kDim + basis] -= diagonal;
        }
        CheckClose(derivative_of(gate), Sandwich(matrix, rho, matrix));
    }
}

TEST_CASE("Dephasing channels lower to diagonal kernels") {
    const parameter::ParameterResolver pr;
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const auto rho = RandomDensityMatrix();
    const auto gamma = 0.3;
    const mqt::CircuitTile tile{{1, 4}, {std::make_shared<mq::PhaseDampingChannel>(gamma, mq::qbits_t{1})}};
    const auto prepared = builder_t::Build(tile, pr, space);
    REQUIRE(prepared.operations.size() == 1);
    CHECK(std::holds_alternative<mqt::DiagonalKernel<value_t>>(prepared.operations.front().kernel));

    auto expected = rho;
    for (size_t row = 0; row < kDim; ++row) {
        for (size_t column = 0; column < kDim; ++column) {
            if (((row ^ column) & 0b010) != 0) {
                expected[row * kDim + column] *= std::sqrt(1 - gamma);
            }
        }
    }
    auto state = rho;
    mqt::detail::ApplyTile<runtime_t>(state.data(), prepared, false);
    CheckClose(state, expected);
}

TEST_CASE("Channels outside their physical range are rejected before lowering") {
    const parameter::ParameterResolver pr;
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const auto lower = [&](const gate_t& gate) {
        const mqt::CircuitTile tile{{1, 4}, {gate}};
        return builder_t::Build(tile, pr, space);
    };
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(lower(std::make_shared<mq::AmplitudeDampingChannel>(false, 1.5, mq::qbits_t{1})),
                    std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::AmplitudeDampingChannel>(false, nan, mq::qbits_t{1})),
                    std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::PhaseDampingChannel>(-0.1, mq::qbits_t{1})), std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::DepolarizingChannel>(1.5, mq::qbits_t{1})), std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::PauliChannel>(0.5, 0.5, 0.5, mq::qbits_t{1})), std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::ThermalRelaxationChannel>(10.0, 30.0, 1.0, mq::qbits_t{1})),
                    std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::ThermalRelaxationChannel>(-1.0, 1.0, 1.0, mq::qbits_t{1})),
                    std::invalid_argument);
    CHECK_THROWS_AS(lower(std::make_shared<mq::AmplitudeDampingChannel>(false, 0.1, mq::qbits_t{1}, mq::qbits_t{0})),
                    std::invalid_argument);
    // The single-qubit channels only act on one qubit.
    CHECK_THROWS_AS(lower(std::make_shared<mq::AmplitudeDampingChannel>(false, 0.1, mq::qbits_t{0, 1})),
                    std::invalid_argument);
    // The depolarizing channel remains completely positive up to 4^m / (4^m - 1).
    CHECK_NOTHROW(lower(std::make_shared<mq::DepolarizingChannel>(4.0 / 3.0, mq::qbits_t{1})));
    CHECK_NOTHROW(lower(std::make_shared<mq::PauliChannel>(0.3, 0.3, 0.4, mq::qbits_t{1})));
}

TEST_CASE("Adjoint lowering inverts unitary evolution") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const mqt::CircuitTile tile{
        {0, 1, 2, 3, 4, 5},
        {std::make_shared<mq::RXGate>(parameter::ParameterResolver{0.37}, mq::qbits_t{1}, mq::qbits_t{0}),
         std::make_shared<mq::FSim>(parameter::ParameterResolver{0.4}, parameter::ParameterResolver{-0.9},
                                    mq::qbits_t{1, 2}, mq::qbits_t{}),
         std::make_shared<mq::ISWAPGate>(true, mq::qbits_t{0, 2}),
         std::make_shared<mq::RotPauliString>("XYZ", parameter::ParameterResolver{0.6}, mq::qbits_t{0, 1, 2})}};
    auto evolved = rho;
    mqt::detail::ApplyTile<runtime_t>(evolved.data(), builder_t::Build(tile, pr, space), false);

    mqt::CircuitTile reversed = tile;
    std::reverse(reversed.gates.begin(), reversed.gates.end());
    const auto adjoint = builder_t::EvolutionOnly(
        builder_t::BuildAdjoint(reversed, pr, {}, space, builder_t::AdjointGates::Reversed));
    mqt::detail::ApplyTile<runtime_t>(evolved.data(), adjoint, false);
    CheckClose(evolved, rho);
}

TEST_CASE("Hermitian mirroring evolves half of the blocks with identical results") {
    const auto rho = RandomDensityMatrix();
    const parameter::ParameterResolver pr;
    const auto space = mqt::StateSpace::DensityMatrix(kQubits);
    const mqt::CircuitTile tile{
        {1, 4},
        {std::make_shared<mq::RXGate>(parameter::ParameterResolver{0.37}, mq::qbits_t{1}, mq::qbits_t{0}),
         std::make_shared<mq::ZGate>(mq::qbits_t{2}), std::make_shared<mq::HGate>(mq::qbits_t{1}),
         std::make_shared<mq::AmplitudeDampingChannel>(false, 0.2, mq::qbits_t{1}),
         std::make_shared<mq::RotPauliString>("ZYZ", parameter::ParameterResolver{0.6}, mq::qbits_t{0, 1, 2})}};
    const auto mirrored = builder_t::Build(tile, pr, space);
    REQUIRE(mirrored.layout.mirror_shift == kQubits);
    auto full = mirrored;
    full.layout.mirror_shift = 0;

    auto mirrored_state = rho;
    auto full_state = rho;
    mqt::detail::ApplyTile<runtime_t>(mirrored_state.data(), mirrored, false);
    mqt::detail::ApplyTile<runtime_t>(full_state.data(), full, false);
    CheckClose(mirrored_state, full_state);

    // Adjoint programs also mirror: overlaps and derivatives of the mirror
    // blocks are evaluated on their conjugate transposes.
    mqt::CircuitTile reversed = tile;
    std::reverse(reversed.gates.begin(), reversed.gates.end());
    const mq::MST<size_t> parameter_map{{"theta", 0}};
    const parameter::ParameterResolver theta(std::string("theta"));
    const mqt::CircuitTile gradient_tile{
        {1, 4},
        {std::make_shared<mq::RXGate>(theta, mq::qbits_t{1}, mq::qbits_t{0}),
         std::make_shared<mq::ZGate>(mq::qbits_t{2}), std::make_shared<mq::HGate>(mq::qbits_t{1}),
         std::make_shared<mq::RYGate>(theta, mq::qbits_t{1})}};
    const parameter::ParameterResolver values(0.0, std::map<std::string, double>{{"theta", 0.41}});
    mqt::AdjointProgram<value_t> program;
    program.output_size = 2;
    program.compute_overlap = true;
    program.tiles.push_back(
        builder_t::BuildAdjoint(gradient_tile, values, parameter_map, space, builder_t::AdjointGates::Reversed));
    auto full_program = program;
    full_program.tiles.front().layout.mirror_shift = 0;

    auto observable = RandomDensityMatrix();
    for (size_t index = 0; index < observable.size(); ++index) {
        observable[index] *= 3.0;
    }
    auto mirrored_shared = rho;
    auto mirrored_observable = observable;
    auto full_shared = rho;
    auto full_observable = observable;
    mq::VVT<value_t> mirrored_output(1, mq::VT<value_t>(2, 0));
    mq::VVT<value_t> full_output(1, mq::VT<value_t>(2, 0));
    mqt::detail::ApplyAdjointProgram<runtime_t>(mirrored_shared.data(), {mirrored_observable.data()}, program,
                                                &mirrored_output, false, false);
    mqt::detail::ApplyAdjointProgram<runtime_t>(full_shared.data(), {full_observable.data()}, full_program,
                                                &full_output, false, false);
    CheckClose(mirrored_shared, full_shared);
    CheckClose(mirrored_observable, full_observable);
    CheckClose(mirrored_output.front(), full_output.front());
    CHECK(std::abs(mirrored_output[0][1]) > 1e-3);
}

TEST_CASE("Density matrix circuits are planned on the doubled index space") {
    const auto space = mqt::StateSpace::DensityMatrix(4);
    const std::vector<gate_t> circuit{
        std::make_shared<mq::HGate>(mq::qbits_t{0}), std::make_shared<mq::XGate>(mq::qbits_t{1}, mq::qbits_t{0}),
        std::make_shared<mq::DepolarizingChannel>(0.1, mq::qbits_t{1}), std::make_shared<mq::ZGate>(mq::qbits_t{3})};
    const auto plan = mqt::Planner(space).Plan(circuit);
    REQUIRE(plan.size() == 1);
    REQUIRE(std::holds_alternative<mqt::CircuitTile>(plan.front()));
    const auto& tile = std::get<mqt::CircuitTile>(plan.front());
    CHECK(tile.gates.size() == circuit.size());
    // The planner keeps the external diagonal operand local while the tile has room.
    CHECK(tile.qubits == mq::qbits_t{0, 1, 3, 4, 5, 7});
}
