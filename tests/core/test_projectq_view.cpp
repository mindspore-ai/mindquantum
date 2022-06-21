//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include <catch2/catch.hpp>
#include <tweedledum/Operators/Standard.h>

#include "core/circuit_manager.hpp"
#include "core/utils.hpp"

// =============================================================================

class UnitTestAccessor {
 public:
    using manager_t = mindquantum::CircuitManager;

    static auto& blocks(const manager_t& manager) {
        return manager.blocks_;
    }
};

using get = UnitTestAccessor;

using mindquantum::CircuitManager;
using mindquantum::committed;
using mindquantum::uncommitted;

// =============================================================================

TEST_CASE("ExternalView/Instructions", "[projectq_view][core]") {
    using instruction_t = CircuitManager::instruction_t;
    using qubit_t = CircuitManager::qubit_t;
    using qubit_array_t = std::vector<qubit_t>;

    CircuitManager manager;

    const auto qubit0 = 10;
    const auto qubit1 = 11;
    const auto qubit2 = 22;

    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit2}));
    REQUIRE(!std::empty(get::blocks(manager)));
    REQUIRE(std::size(get::blocks(manager)) == 1);

    using op1_t = tweedledum::Op::X;
    using op2_t = tweedledum::Op::H;
    using op3_t = tweedledum::Op::S;
    using op4_t = tweedledum::Op::T;

    manager.apply_operator(op1_t(), {}, {qubit0});
    manager.apply_operator(op2_t(), {qubit1}, {qubit0});
    manager.apply_operator(op3_t(), {qubit0}, {qubit1});

    auto view(manager.as_projectq(committed));

    std::vector<instruction_t> instructions;

    SECTION("Empty") {
        // NB: these are iterating over committed operations...
        view.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });
        CHECK(std::empty(instructions));

        view.foreach_r_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });
        CHECK(std::empty(instructions));
    }

    manager.commit_changes();
    manager.apply_operator(op4_t(), {}, {qubit1});
    manager.commit_changes();

    REQUIRE(std::size(get::blocks(manager)) == 3);

    SECTION("foreach_instructions") {
        view.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 4);
        CHECK(instructions[0].kind() == op1_t::kind());
        CHECK(instructions[0].qubits() == qubit_array_t{qubit_t(qubit0)});
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == qubit_array_t{qubit_t(qubit1), qubit_t(qubit0)});
        CHECK(instructions[2].kind() == op3_t::kind());
        CHECK(instructions[2].qubits() == qubit_array_t{qubit_t(qubit0), qubit_t(qubit1)});
        CHECK(instructions[3].kind() == op4_t::kind());
        CHECK(instructions[3].qubits() == qubit_array_t{qubit_t(qubit1)});
    }

    SECTION("foreach_r_instructions") {
        view.foreach_r_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 4);
        CHECK(instructions[0].kind() == op4_t::kind());
        CHECK(instructions[0].qubits() == qubit_array_t{qubit_t(qubit1)});
        CHECK(instructions[1].kind() == op3_t::kind());
        CHECK(instructions[1].qubits() == qubit_array_t{qubit_t(qubit0), qubit_t(qubit1)});
        CHECK(instructions[2].kind() == op2_t::kind());
        CHECK(instructions[2].qubits() == qubit_array_t{qubit_t(qubit1), qubit_t(qubit0)});
        CHECK(instructions[3].kind() == op1_t::kind());
        CHECK(instructions[3].qubits() == qubit_array_t{qubit_t(qubit0)});
    }
}

// =============================================================================

TEST_CASE("ExternalBlockView/Instructions", "[projectq_view][core]") {
    using instruction_t = CircuitManager::instruction_t;
    using qubit_t = CircuitManager::qubit_t;
    using qubit_array_t = std::vector<qubit_t>;

    CircuitManager manager;

    const auto qubit0 = 10;
    const auto qubit1 = 11;
    const auto qubit2 = 22;

    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit2}));
    REQUIRE(!std::empty(get::blocks(manager)));
    REQUIRE(std::size(get::blocks(manager)) == 1);

    using op1_t = tweedledum::Op::X;
    using op2_t = tweedledum::Op::H;
    using op3_t = tweedledum::Op::S;

    manager.apply_operator(tweedledum::Op::T(), {}, {qubit0});
    manager.commit_changes();

    manager.apply_operator(op1_t(), {}, {qubit0});
    manager.apply_operator(op2_t(), {qubit1}, {qubit0});
    manager.apply_operator(op3_t(), {qubit0}, {qubit1});

    auto view(manager.as_projectq(uncommitted));

    std::vector<instruction_t> instructions;

    REQUIRE(std::size(get::blocks(manager)) == 2);

    SECTION("foreach_instructions") {
        view.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 3);
        CHECK(instructions[0].kind() == op1_t::kind());
        CHECK(instructions[0].qubits() == qubit_array_t{qubit_t(qubit0)});
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == qubit_array_t{qubit_t(qubit1), qubit_t(qubit0)});
        CHECK(instructions[2].kind() == op3_t::kind());
        CHECK(instructions[2].qubits() == qubit_array_t{qubit_t(qubit0), qubit_t(qubit1)});
    }

    SECTION("foreach_r_instructions") {
        view.foreach_r_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 3);
        CHECK(instructions[0].kind() == op3_t::kind());
        CHECK(instructions[0].qubits() == qubit_array_t{qubit_t(qubit0), qubit_t(qubit1)});
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == qubit_array_t{qubit_t(qubit1), qubit_t(qubit0)});
        CHECK(instructions[2].kind() == op1_t::kind());
        CHECK(instructions[2].qubits() == qubit_array_t{qubit_t(qubit0)});
    }
}

// =============================================================================
