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
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <variant>
#include <vector>

#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "simulator/circuit_tiling/builder.h"
#include "simulator/circuit_tiling/executor.h"
#include "simulator/circuit_tiling/planner.h"
#include "simulator/cpu_features.h"
#include "simulator/worker_scratch.h"
#if MQ_HAS_ARM_SVE_BACKEND
#    include "simulator/test_circuit_tiling_sve.h"
#endif

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

TEST_CASE("Circuit tile scratch rejects overflowing sizes without damaging worker storage") {
    const auto check = [](auto value) {
        using calc_type = decltype(value);
        mq::sim::WorkerScratch<calc_type> scratch;
        CHECK_NOTHROW(mqt::ReserveScratch(scratch, 0, 3));
        mqt::ReserveScratch(scratch, 7, 3);
        for (size_t worker = 0; worker < 3; ++worker) {
            std::fill_n(scratch.ForWorker(worker), 14, static_cast<calc_type>(worker + 1));
        }

        constexpr auto maximum = std::numeric_limits<size_t>::max();
        for (auto size : {maximum / sizeof(calc_type) / 2 + 1, maximum / 2 + 1, maximum}) {
            CAPTURE(sizeof(calc_type), size);
            CHECK_THROWS_AS(mqt::ReserveScratch(scratch, size, 3), std::bad_array_new_length);
        }
        for (size_t worker = 0; worker < 3; ++worker) {
            for (size_t element = 0; element < 14; ++element) {
                CHECK(scratch.ForWorker(worker)[element] == static_cast<calc_type>(worker + 1));
            }
        }
    };
    check(float{});
    check(double{});
}

TEST_CASE("Circuit tile execution preserves external controls") {
    const auto check = [](auto value) {
        using calc_type = decltype(value);
        using value_t = std::complex<calc_type>;

        // Cover uncontrolled, intra-vector per-lane ({1}), block-wide ({8}, {11}),
        // and combined multi-qubit external control masks.
        for (const auto& controls : {mq::qbits_t{}, mq::qbits_t{1}, mq::qbits_t{8}, mq::qbits_t{11}, mq::qbits_t{1, 8},
                                     mq::qbits_t{2, 8, 11}}) {
            mqt::CircuitTile tile;
            tile.qubits = {0};
            tile.gates = {std::make_shared<mq::SGate>(mq::qbits_t{0}, controls)};
            const auto prepared = mqt::Builder<value_t>::Build(tile, parameter::ParameterResolver{}, VectorSpace(12));
            mq::index_t required = 1;  // S acts on target |1>, with every control set.
            for (const auto control : controls) {
                required |= mq::index_t{1} << control;
            }
            for (const int threads : {1, 2}) {
                CAPTURE(sizeof(calc_type), controls, threads);
                std::vector<value_t> state(1U << 12), expected(state.size());
                for (size_t index = 0; index < state.size(); ++index) {
                    const auto real = static_cast<calc_type>(index % 17 + 2);
                    const auto imaginary = static_cast<calc_type>(index % 13) - 4;
                    state[index] = {real, imaginary};
                    expected[index] = (index & required) == required ? value_t{-imaginary, real} : state[index];
                }
                mq::sim::WorkerScratch<calc_type> scratch;
                mqt::ReserveScratch(scratch, prepared.layout.ScratchSize(), threads);
                mqt::ApplyTile(state.data(), prepared, threads, scratch);
                // Multiplication by i is exact: check active and inactive amplitudes
                // against gate semantics, independently of the execution backend.
                CHECK(std::equal(state.begin(), state.end(), expected.begin()));
            }
        }

        // Small spaces exercise partial vector predicates where trailing lanes are
        // inactive; trailing canaries verify inactive lanes never scatter past state.
        constexpr size_t kCanaryCount = 64;
        const value_t canary{calc_type{-97}, calc_type{89}};
        for (const mq::qbit_t n_qubits : {1, 2, 3}) {
            const std::vector<mq::qbits_t> control_sets = n_qubits == 1
                                                              ? std::vector<mq::qbits_t>{{}}
                                                              : (n_qubits == 2
                                                                     ? std::vector<mq::qbits_t>{{}, {1}}
                                                                     : std::vector<mq::qbits_t>{{1}, {2}, {1, 2}});
            for (const auto& controls : control_sets) {
                mqt::CircuitTile tile;
                tile.qubits = {0};
                tile.gates = {std::make_shared<mq::SGate>(mq::qbits_t{0}, controls)};
                const auto prepared = mqt::Builder<value_t>::Build(tile, parameter::ParameterResolver{},
                                                                   VectorSpace(n_qubits));
                mq::index_t required = 1;
                for (const auto control : controls) {
                    required |= mq::index_t{1} << control;
                }
                const size_t dim = size_t{1} << n_qubits;
                std::vector<value_t> buffer(dim + kCanaryCount, canary), expected(dim + kCanaryCount, canary);
                for (size_t index = 0; index < dim; ++index) {
                    const auto real = static_cast<calc_type>(index + 3);
                    const auto imaginary = static_cast<calc_type>(index) - calc_type{2};
                    buffer[index] = {real, imaginary};
                    expected[index] = (index & required) == required ? value_t{-imaginary, real} : buffer[index];
                }
                CAPTURE(sizeof(calc_type), n_qubits, controls);
                mq::sim::WorkerScratch<calc_type> scratch;
                mqt::ReserveScratch(scratch, prepared.layout.ScratchSize(), 1);
                mqt::ApplyTile(buffer.data(), prepared, 1, scratch);
                CHECK(std::equal(buffer.begin(), buffer.end(), expected.begin()));
            }
        }

        // Sequential operations in a single tile must each evaluate their own
        // external controls without clobbering the shared active predicate.
        mqt::CircuitTile multi_op_tile;
        multi_op_tile.qubits = {0};
        multi_op_tile.gates = {std::make_shared<mq::XGate>(mq::qbits_t{0}, mq::qbits_t{1}),
                               std::make_shared<mq::SGate>(mq::qbits_t{0}, mq::qbits_t{8}),
                               std::make_shared<mq::ZGate>(mq::qbits_t{0}, mq::qbits_t{1, 11}),
                               std::make_shared<mq::SGate>(mq::qbits_t{0})};
        const auto prepared_multi = mqt::Builder<value_t>::Build(multi_op_tile, parameter::ParameterResolver{},
                                                                 VectorSpace(12));
        for (const int threads : {1, 2}) {
            CAPTURE(sizeof(calc_type), threads);
            std::vector<value_t> state(1U << 12), expected(state.size());
            for (size_t index = 0; index < state.size(); ++index) {
                state[index] = {static_cast<calc_type>(index % 19 + 1), static_cast<calc_type>(index % 11) - 5};
            }
            expected = state;
            for (size_t base = 0; base < expected.size(); base += 2) {
                if ((base & (size_t{1} << 1)) != 0) {
                    std::swap(expected[base], expected[base + 1]);
                }
                if ((base & (size_t{1} << 8)) != 0) {
                    expected[base + 1] = {-expected[base + 1].imag(), expected[base + 1].real()};
                }
                if ((base & ((size_t{1} << 1) | (size_t{1} << 11))) == ((size_t{1} << 1) | (size_t{1} << 11))) {
                    expected[base + 1] = -expected[base + 1];
                }
                expected[base + 1] = {-expected[base + 1].imag(), expected[base + 1].real()};
            }
            mq::sim::WorkerScratch<calc_type> scratch;
            mqt::ReserveScratch(scratch, prepared_multi.layout.ScratchSize(), threads);
            mqt::ApplyTile(state.data(), prepared_multi, threads, scratch);
            CHECK(std::equal(state.begin(), state.end(), expected.begin()));
        }
    };
    check(float{});
    check(double{});

#if MQ_HAS_ARM_SVE_BACKEND
    if (mq::sim::cpu::HostHasArmSveFcma()) {
        const auto report = mqt::test::VerifySveExternalControlWindows();
        INFO(report.failure);
        CHECK(report.ok);
    }
#endif
}

TEST_CASE("Adjoint tile execution preserves external controls") {
    const auto check = [](auto value) {
        using calc_type = decltype(value);
        using value_t = std::complex<calc_type>;
        using builder_t = mqt::Builder<value_t>;

        const parameter::ParameterResolver theta(std::string("theta"));
        const parameter::ParameterResolver values(0.0, std::map<std::string, double>{{"theta", 0.0}});
        const mq::MST<size_t> parameter_map{{"theta", 0}};
        const auto space = VectorSpace(8);
        constexpr size_t kDim = 1U << 8;

        for (const auto& controls : {mq::qbits_t{}, mq::qbits_t{1}, mq::qbits_t{5}, mq::qbits_t{1, 7}}) {
            mqt::CircuitTile tile;
            tile.qubits = {0};
            tile.gates = {std::make_shared<mq::SGate>(mq::qbits_t{0}, controls),
                          std::make_shared<mq::PSGate>(theta, mq::qbits_t{0}, controls)};

            mqt::AdjointProgram<value_t> program;
            program.output_size = 2;
            program.compute_overlap = true;
            program.tiles.push_back(
                builder_t::BuildAdjoint(tile, values, parameter_map, space, builder_t::AdjointGates::Daggered));

            mq::index_t required = 1;
            for (const auto control : controls) {
                required |= mq::index_t{1} << control;
            }

            // Small integer amplitudes keep every overlap and derivative sum
            // well within IEEE-754 float's 24-bit exact integer range, making
            // reductions exact regardless of vector width or thread count.
            std::vector<value_t> initial_bra(kDim), initial_ket(kDim);
            std::vector<value_t> expected_bra(kDim), expected_ket(kDim);
            value_t raw_overlap{0, 0};
            value_t raw_derivative{0, 0};
            for (size_t index = 0; index < kDim; ++index) {
                const value_t bra_amp{static_cast<calc_type>(index % 5) - calc_type{2},
                                      static_cast<calc_type>((index / 3) % 5) - calc_type{2}};
                const value_t ket_amp{static_cast<calc_type>((index + 1) % 7) - calc_type{3},
                                      static_cast<calc_type>((index / 2) % 5) - calc_type{2}};
                initial_bra[index] = bra_amp;
                initial_ket[index] = ket_amp;
                raw_overlap += std::conj(bra_amp) * ket_amp;
                if ((index & required) == required) {
                    expected_bra[index] = {-bra_amp.imag(), bra_amp.real()};
                    expected_ket[index] = {-ket_amp.imag(), ket_amp.real()};
                    raw_derivative += std::conj(expected_bra[index])
                                      * value_t{-expected_ket[index].imag(), expected_ket[index].real()};
                } else {
                    expected_bra[index] = bra_amp;
                    expected_ket[index] = ket_amp;
                }
            }

            for (const bool conjugate_output : {false, true}) {
                for (const int threads : {1, 2}) {
                    CAPTURE(sizeof(calc_type), controls, conjugate_output, threads);
                    auto bra = initial_bra;
                    auto ket = initial_ket;
                    mq::VVT<value_t> output(1, mq::VT<value_t>(2, 0));
                    mq::sim::WorkerScratch<calc_type> scratch;
                    mqt::ReserveScratch(scratch, program.tiles.front().layout.ScratchSize(2), threads);
                    mqt::ApplyAdjointProgram(bra.data(), {ket.data()}, program, &output, conjugate_output, threads,
                                             scratch);
                    const auto expected_overlap = conjugate_output ? std::conj(raw_overlap) : raw_overlap;
                    const auto expected_deriv = conjugate_output ? std::conj(raw_derivative) : raw_derivative;
                    CHECK(output[0][0] == expected_overlap);
                    CHECK(output[0][1] == expected_deriv);
                    CHECK(std::equal(bra.begin(), bra.end(), expected_bra.begin()));
                    CHECK(std::equal(ket.begin(), ket.end(), expected_ket.begin()));
                }
            }
        }
    };
    check(float{});
    check(double{});
}

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

TEST_CASE("Candidate rollback follows live degrees across many tile commits") {
    constexpr size_t rounds = 12;
    circuit_t circuit;
    mqt::Schedule expected;
    std::vector<mq::index_t> mixing_masks;
    std::vector<mq::index_t> support_masks;
    for (size_t round = 0; round < rounds; ++round) {
        const auto first = static_cast<mq::qbit_t>(2 + round % 3);
        const auto second = static_cast<mq::qbit_t>(2 + (round + 1) % 3);
        const auto third = static_cast<mq::qbit_t>(2 + (round + 2) % 3);
        const auto base = circuit.size();
        circuit.insert(
            circuit.end(),
            {std::make_shared<mq::SWAPGate>(mq::qbits_t{0, 1}),
             std::make_shared<mq::XGate>(mq::qbits_t{first}, mq::qbits_t{0}),
             std::make_shared<mq::ZGate>(mq::qbits_t{0}),
             std::make_shared<mq::XGate>(mq::qbits_t{second}, mq::qbits_t{0}),
             std::make_shared<mq::XGate>(mq::qbits_t{third}, mq::qbits_t{0}),
             std::make_shared<mq::ZGate>(mq::qbits_t{first}), std::make_shared<mq::ZGate>(mq::qbits_t{second}),
             std::make_shared<mq::ZGate>(mq::qbits_t{third}), std::make_shared<mq::SGate>(mq::qbits_t{third})});

        // Independent access oracle: SWAP/X mix their targets; Z/S and controls
        // only read. Do not reuse the planner's gate descriptions or graph.
        const auto a = mq::index_t{1} << first;
        const auto b = mq::index_t{1} << second;
        const auto c = mq::index_t{1} << third;
        mixing_masks.insert(mixing_masks.end(), {3, a, 0, b, c, 0, 0, 0, 0});
        support_masks.insert(support_masks.end(), {3, a | 1, 1, b | 1, c | 1, a, b, c, c});

        // Hand-derived boundaries with two mixing bits: SWAP(0,1) fills its
        // tile and releases three branches plus the commuting Z(0). A pair
        // containing the third branch wins (five gates versus four); the tie
        // goes to the first branch's earlier gate, not the lower qubit mask.
        // The remaining branch needs its own tile, with room to localize its
        // control. All three controls and Z(0) feed the next SWAP, which cannot
        // share a tile with any branch target. There are no segment barriers:
        // the same evaluator must survive all 36 tile commits.
        expected.emplace_back(mqt::CircuitTile{{0, 1}, {circuit[base], circuit[base + 2]}});
        expected.emplace_back(mqt::CircuitTile{
            {std::min(first, third), std::max(first, third)},
            {circuit[base + 1], circuit[base + 4], circuit[base + 5], circuit[base + 7], circuit[base + 8]}});
        expected.emplace_back(mqt::CircuitTile{{0, second}, {circuit[base + 3], circuit[base + 6]}});
    }

    const mqt::Planner planner(VectorSpace(5), 2);
    const auto plan = planner.Plan(circuit);
    REQUIRE(plan.size() == 3 * rounds);
    CheckSamePlan(expected, plan);

    const auto flattened = Flatten(plan);
    REQUIRE(flattened.size() == circuit.size());
    REQUIRE(std::is_permutation(flattened.begin(), flattened.end(), circuit.begin(), circuit.end()));
    std::vector<size_t> positions;
    for (const auto& gate : circuit) {
        positions.push_back(Position(flattened, gate));
    }
    // Every pair with a mixing/read or mixing/mixing conflict must retain its
    // input order, including dependencies spanning multiple rounds.
    for (size_t before = 0; before < circuit.size(); ++before) {
        for (size_t after = before + 1; after < circuit.size(); ++after) {
            if ((mixing_masks[before] & support_masks[after]) != 0
                || (support_masks[before] & mixing_masks[after]) != 0) {
                CAPTURE(before, after);
                CHECK(positions[before] < positions[after]);
            }
        }
    }
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
