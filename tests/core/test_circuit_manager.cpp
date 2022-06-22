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

#include "core/circuit_block.hpp"
#include "core/circuit_manager.hpp"
#include "core/utils.hpp"
#include "ops/gates/measure.hpp"

namespace td = tweedledum;

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

TEST_CASE("CircuitManager/Construction", "[circuit_manager][core]") {
    CircuitManager manager;
    REQUIRE(!std::empty(get::blocks(manager)));
    REQUIRE(std::size(get::blocks(manager)) == 1);
}

// =============================================================================

TEST_CASE("CircuitManager/commit_changes", "[circuit_manager][core]") {
    CircuitManager manager;
    REQUIRE(std::size(get::blocks(manager)) == 1);

    manager.commit_changes();
    REQUIRE(std::size(get::blocks(manager)) == 2);

    manager.commit_changes();
    REQUIRE(std::size(get::blocks(manager)) == 3);
}

// -----------------------------------------------------------------------------

TEST_CASE("CircuitManager/Add qubits", "[circuit_manager][core]") {
    CircuitManager manager;

    CHECK(std::size(manager) == 0);

    SECTION("Add qubit") {
        const auto qubit0 = mindquantum::QubitID{10};
        const auto qubit1 = mindquantum::QubitID{11};

        CHECK(manager.add_qubit(qubit0));
        CHECK(!manager.add_qubit(qubit0));

        CHECK(manager.has_qubit(qubit0));
        CHECK(!manager.has_qubit(qubit1));

        CHECK(std::size(manager) == 0);
    }

    SECTION("Translate IDs") {
        const auto qubit0 = 10;
        const auto qubit1 = 11;
        REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit0}));
        REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit1}));

        CHECK(manager.translate_id(td::Qubit(0)) == qubit0);
        CHECK(manager.translate_id(td::Qubit(1)) == qubit1);
    }
}

// -----------------------------------------------------------------------------

TEST_CASE("CircuitManager/Apply operators", "[circuit_manager][core]") {
    using qubit_t = CircuitManager::qubit_t;
    using instruction_t = CircuitManager::instruction_t;
    using block_t = mindquantum::CircuitBlock;

    CircuitManager manager;
    const auto qubit0 = 10;
    const auto qubit1 = 11;

    REQUIRE(std::size(get::blocks(manager)) == 1);
    REQUIRE(std::size(manager) == 0);
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(manager.add_qubit(mindquantum::QubitID{qubit1}));

    SECTION("Single block") {
        using op_t = tweedledum::Op::X;
        manager.apply_operator(op_t(), {}, {qubit0});

        CHECK(std::size(manager) == 1);
        CHECK(manager.size(committed) == 0);
        CHECK(manager.size(uncommitted) == 1);

        std::vector<const instruction_t*> instructions;
        get::blocks(manager)[0].foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });
        CHECK(std::size(instructions) == 1);
        CHECK(instructions[0]->kind() == op_t::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
    }

    SECTION("Measurement") {
        manager.apply_measurement(qubit0);

        CHECK(std::size(manager) == 1);
        CHECK(manager.size(committed) == 0);
        CHECK(manager.size(uncommitted) == 1);

        std::vector<const instruction_t*> instructions;
        get::blocks(manager)[0].foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });
        CHECK(std::size(instructions) == 1);
        CHECK(instructions[0]->kind() == mindquantum::ops::Measure::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
    }

    SECTION("Multiple block") {
        using op1_t = tweedledum::Op::X;
        using op2_t = tweedledum::Op::H;
        using op3_t = tweedledum::Op::S;

        manager.apply_operator(op1_t(), {}, {qubit0});
        CHECK(std::size(manager) == 1);
        CHECK(manager.size(committed) == 0);
        CHECK(manager.size(uncommitted) == 1);

        manager.commit_changes();
        CHECK(std::size(manager) == 1);
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 0);

        manager.apply_operator(op2_t(), {}, {qubit0});
        manager.apply_operator(op3_t(), {qubit0}, {qubit1});

        CHECK(std::size(manager) == 3);
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 2);

        manager.commit_changes();
        CHECK(std::size(manager) == 3);
        CHECK(manager.size(committed) == 3);
        CHECK(manager.size(uncommitted) == 0);

        manager.apply_operator(tweedledum::Op::T(), {qubit0}, {qubit1});

        std::vector<const instruction_t*> instructions, instructions_r;
        manager.foreach_block([&instructions](const block_t& block) {
            block.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });
        });
        manager.foreach_r_block([&instructions_r](const block_t& block) {
            block.foreach_r_instruction(
                [&instructions_r](const instruction_t& inst) { instructions_r.emplace_back(&inst); });
        });

        REQUIRE(std::size(instructions) == 3);  // Last T gate should not be present! (uncommitted)
        CHECK(instructions[0]->kind() == op1_t::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[1]->kind() == op2_t::kind());
        CHECK(instructions[1]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[2]->kind() == op3_t::kind());
        CHECK(instructions[2]->qubits() == std::vector<qubit_t>{qubit_t(0), qubit_t(1)});

        std::reverse(std::begin(instructions_r), std::end(instructions_r));
        CHECK_THAT(instructions, Equals(instructions_r));
    }

    SECTION("Foreach block") {
        using op1_t = tweedledum::Op::X;
        using op2_t = tweedledum::Op::H;
        using op3_t = tweedledum::Op::S;

        manager.apply_operator(op1_t(), {}, {qubit0});
        CHECK(manager.size(committed) == 0);
        CHECK(manager.size(uncommitted) == 1);

        manager.commit_changes();
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 0);

        manager.apply_operator(op2_t(), {}, {qubit0});
        manager.apply_operator(op3_t(), {qubit0}, {qubit1});

        CHECK(std::size(manager) == 3);
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 2);

        std::vector<const instruction_t*> instructions;
        manager.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); },
                                    uncommitted);

        REQUIRE(std::size(instructions) == 2);  // Last T gate should not be present! (uncommitted)
        CHECK(instructions[0]->kind() == op2_t::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[1]->kind() == op3_t::kind());
        CHECK(instructions[1]->qubits() == std::vector<qubit_t>{qubit_t(0), qubit_t(1)});
    }
}

// =============================================================================

TEST_CASE("CircuitManager/Delete qubits", "[circuit_manager][core]") {
    using qubit_t = CircuitManager::qubit_t;
    using instruction_t = CircuitManager::instruction_t;
    using block_t = mindquantum::CircuitBlock;

    CircuitManager manager;
    const auto qubit0 = mindquantum::QubitID{10};
    const auto qubit1 = mindquantum::QubitID{11};

    REQUIRE(std::size(get::blocks(manager)) == 1);
    REQUIRE(std::size(manager) == 0);
    REQUIRE(manager.add_qubit(qubit0));
    REQUIRE(manager.add_qubit(qubit1));

    SECTION("Empty circuit") {
        CHECK(std::size(get::blocks(manager)) == 1);
        CHECK(std::size(manager) == 0);

        manager.delete_qubits({qubit0});

        CHECK(std::size(get::blocks(manager)) == 1);

        CHECK(!manager.has_qubit(qubit0));
        CHECK(manager.has_qubit(qubit1));

        manager.delete_qubits({qubit0});

        CHECK(std::size(get::blocks(manager)) == 1);

        CHECK(!manager.has_qubit(qubit0));
        CHECK(manager.has_qubit(qubit1));
    }
    SECTION("Empty last block") {
        CHECK(std::size(get::blocks(manager)) == 1);

        manager.apply_operator(tweedledum::Op::X(), {}, {static_cast<unsigned int>(qubit0)});
        manager.commit_changes();

        CHECK(std::size(get::blocks(manager)) == 2);
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 0);

        manager.delete_qubits({qubit1});

        CHECK(std::size(get::blocks(manager)) == 2);
        CHECK(manager.has_qubit(qubit0));
        CHECK(!manager.has_qubit(qubit1));
    }

    SECTION("Non empty last block") {
        manager.apply_operator(tweedledum::Op::X(), {}, {static_cast<unsigned int>(qubit0)});
        manager.commit_changes();
        manager.apply_operator(tweedledum::Op::Y(), {static_cast<unsigned int>(qubit0)},
                               {static_cast<unsigned int>(qubit1)});

        CHECK(std::size(get::blocks(manager)) == 2);
        CHECK(manager.size(committed) == 1);
        CHECK(manager.size(uncommitted) == 1);

        manager.delete_qubits({qubit0});

        CHECK(std::size(get::blocks(manager)) == 3);
        CHECK(!manager.has_qubit(qubit0));
        CHECK(manager.has_qubit(qubit1));

        manager.delete_qubits({qubit0});

        CHECK(std::size(get::blocks(manager)) == 3);
        CHECK(!manager.has_qubit(qubit0));
        CHECK(manager.has_qubit(qubit1));

        manager.delete_qubits({qubit1});

        CHECK(std::size(get::blocks(manager)) == 3);
        CHECK(!manager.has_qubit(qubit0));
        CHECK(!manager.has_qubit(qubit1));
    }
}

// =============================================================================
