//   Copyright 2021 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Target/Placement.h>

#include "mindquantum/catch2/mindquantum.hpp"

#include "experimental/mapping/partial_placer.hpp"

#include <catch2/catch_test_macros.hpp>

// =============================================================================

using mindquantum::mapping::PartialPlacer;

using qubit_t = tweedledum::Qubit;
using device_t = tweedledum::Device;
using circuit_t = tweedledum::Circuit;
using placement_t = tweedledum::Placement;

// =============================================================================

TEST_CASE("PartialPlacer/No cbits", "[mapping][placer]") {
    circuit_t circuit;
    qubit_t q[] = {
        circuit.create_qubit(),
        circuit.create_qubit(),
        circuit.create_qubit(),
        circuit.create_qubit(),
    };

    device_t device = device_t::ring(circuit.num_qubits());
    placement_t placement(device.num_qubits(), circuit.num_qubits());

    SECTION("Place single qubit") {
        for (auto i(0UL); i < std::size(q) - 1; ++i) {
            placement.map_v_phy(q[i], q[i]);
        }

        auto v_to_phy_ref = decltype(placement.v_to_phy()){
            qubit_t(0),
            qubit_t(1),
            qubit_t(2),
            qubit_t::invalid(),
        };
        REQUIRE(placement.v_to_phy() == v_to_phy_ref);

        auto phy_to_v_ref = decltype(placement.phy_to_v()){
            qubit_t(0),
            qubit_t(1),
            qubit_t(2),
            qubit_t::invalid(),
        };
        REQUIRE(placement.phy_to_v() == phy_to_v_ref);

        PartialPlacer placer(device, placement);
        placer.run({q[3]});

        v_to_phy_ref.back() = qubit_t(3);
        CHECK(placement.v_to_phy() == v_to_phy_ref);

        phy_to_v_ref.back() = qubit_t(3);
        CHECK(placement.phy_to_v() == phy_to_v_ref);
    }

    SECTION("Place two qubits") {
        for (auto i(0UL); i < std::size(q) - 2; ++i) {
            placement.map_v_phy(q[i], q[i]);
        }

        auto v_to_phy_ref = decltype(placement.v_to_phy()){
            qubit_t(0),
            qubit_t(1),
            qubit_t::invalid(),
            qubit_t::invalid(),
        };
        REQUIRE(placement.v_to_phy() == v_to_phy_ref);

        auto phy_to_v_ref = decltype(placement.phy_to_v()){
            qubit_t(0),
            qubit_t(1),
            qubit_t::invalid(),
            qubit_t::invalid(),
        };
        REQUIRE(placement.phy_to_v() == phy_to_v_ref);

        PartialPlacer placer(device, placement);
        placer.run({q[2], q[3]});

        v_to_phy_ref[2] = qubit_t(3);
        v_to_phy_ref[3] = qubit_t(2);
        CHECK(placement.v_to_phy() == v_to_phy_ref);

        phy_to_v_ref[2] = qubit_t(3);
        phy_to_v_ref[3] = qubit_t(2);
        CHECK(placement.phy_to_v() == phy_to_v_ref);
    }

    SECTION("Place three qubits") {
        placement.map_v_phy(q[0], q[0]);

        auto v_to_phy_ref = decltype(placement.v_to_phy()){
            qubit_t(0),
            qubit_t::invalid(),
            qubit_t::invalid(),
            qubit_t::invalid(),
        };
        REQUIRE(placement.v_to_phy() == v_to_phy_ref);

        auto phy_to_v_ref = decltype(placement.phy_to_v()){
            qubit_t(0),
            qubit_t::invalid(),
            qubit_t::invalid(),
            qubit_t::invalid(),
        };
        REQUIRE(placement.phy_to_v() == phy_to_v_ref);

        PartialPlacer placer(device, placement);
        placer.run({q[2], q[1], q[3]});

        v_to_phy_ref[1] = qubit_t(2);
        v_to_phy_ref[2] = qubit_t(3);
        v_to_phy_ref[3] = qubit_t(1);
        CHECK(placement.v_to_phy() == v_to_phy_ref);

        phy_to_v_ref[1] = qubit_t(3);
        phy_to_v_ref[2] = qubit_t(1);
        phy_to_v_ref[3] = qubit_t(2);
        CHECK(placement.phy_to_v() == phy_to_v_ref);
    }
}

// =============================================================================

TEST_CASE("PartialPlacer/No cbits + add v qubits", "[mapping][placer]") {
    circuit_t circuit;
    qubit_t q[] = {circuit.create_qubit(), circuit.create_qubit(), qubit_t::invalid(), qubit_t::invalid()};

    device_t device = device_t::ring(std::size(q));
    placement_t placement(device.num_qubits(), circuit.num_qubits());

    for (auto i(0UL); i < circuit.num_qubits(); ++i) {
        placement.map_v_phy(q[i], q[i]);
    }

    INFO("OK1");

    CHECK(size(placement.v_to_phy()) == circuit.num_qubits());
    CHECK(size(placement.phy_to_v()) == device.num_qubits());

    INFO("OK2");
    auto v_to_phy_ref = decltype(placement.v_to_phy()){
        qubit_t(0),
        qubit_t(1),
    };
    REQUIRE(placement.v_to_phy() == v_to_phy_ref);
    INFO("OK3");

    auto phy_to_v_ref = decltype(placement.phy_to_v()){
        qubit_t(0),
        qubit_t(1),
        qubit_t::invalid(),
        qubit_t::invalid(),
    };
    REQUIRE(placement.phy_to_v() == phy_to_v_ref);

    INFO("OK4");
    q[2] = circuit.create_qubit();
    q[3] = circuit.create_qubit();

    INFO("OK6");
    PartialPlacer placer(device, placement);
    placer.run({q[2], q[3]});
    INFO("OK5");

    v_to_phy_ref.emplace_back(qubit_t{3});
    v_to_phy_ref.emplace_back(qubit_t{2});
    CHECK(placement.v_to_phy() == v_to_phy_ref);
    INFO("OK6");

    assert(size(phy_to_v_ref) == 4);
    phy_to_v_ref.at(2) = qubit_t(3);
    phy_to_v_ref.at(3) = qubit_t(2);
    INFO("OK7");
    // CHECK(placement.phy_to_v() == phy_to_v_ref);
}

// =============================================================================

// This test case is now redundant since qubits and cbits are stored separately
TEST_CASE("PartialPlacer/With cbits", "[mapping][placer]") {
    circuit_t circuit;

    qubit_t q[4] = {
        qubit_t::invalid(),
        qubit_t::invalid(),
        qubit_t::invalid(),
        qubit_t::invalid(),
    };

    q[0] = circuit.create_qubit();
    auto c0 = circuit.create_cbit();
    q[1] = circuit.create_qubit();
    q[2] = circuit.create_qubit();
    auto c1 = circuit.create_cbit();
    auto c2 = circuit.create_cbit();
    q[3] = circuit.create_qubit();

    device_t device = device_t::ring(circuit.num_qubits());
    placement_t placement(device.num_qubits(), circuit.num_qubits());

    placement.map_v_phy(q[0], qubit_t(2));

    auto v_to_phy_ref = decltype(placement.v_to_phy()){
        qubit_t(2),
        qubit_t::invalid(),
        qubit_t::invalid(),
        qubit_t::invalid(),
    };
    REQUIRE(placement.v_to_phy() == v_to_phy_ref);

    auto phy_to_v_ref = decltype(placement.phy_to_v()){
        qubit_t::invalid(),
        qubit_t::invalid(),
        qubit_t(0),
        qubit_t::invalid(),
    };
    REQUIRE(placement.phy_to_v() == phy_to_v_ref);

    PartialPlacer placer(device, placement);
    placer.run({q[1], q[2], q[3]});

    v_to_phy_ref[1] = qubit_t(3);
    v_to_phy_ref[2] = qubit_t(1);
    v_to_phy_ref[3] = qubit_t(0);
    CHECK(placement.v_to_phy() == v_to_phy_ref);

    phy_to_v_ref[0] = qubit_t(3);
    phy_to_v_ref[1] = qubit_t(2);
    phy_to_v_ref[3] = qubit_t(1);
    CHECK(placement.phy_to_v() == phy_to_v_ref);
}

// =============================================================================
