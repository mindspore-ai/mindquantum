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

#if MQ_HAS_CONCEPTS
#    include <concepts>
#endif  // MQ_HAS_CONCEPTS
#include <type_traits>

#include <catch2/catch.hpp>
#include <tweedledum/Operators/Standard.h>
#include <tweedledum/Passes/Mapping/Placer/TrivialPlacer.h>

#include "core/circuit_manager.hpp"
#include "core/utils.hpp"

namespace td = tweedledum;

// =============================================================================

class UnitTestAccessor {
 public:
    using block_t = mindquantum::CircuitBlock;
    using manager_t = mindquantum::CircuitManager;

    static auto& blocks(manager_t& manager) {
        return manager.blocks_;
    }

    static auto& device(block_t& block) {
        return block.device_;
    }

    static auto& mapping(block_t& block) {
        return block.mapping_;
    }

    static auto& initial_mapping(block_t& block) {
        return block.mapping_.value().init_placement;
    }

    static auto& final_mapping(block_t& block) {
        return block.mapping_.value().placement;
    }
};

using get = UnitTestAccessor;

using mindquantum::CircuitManager;
using mindquantum::committed;
using mindquantum::uncommitted;

// =============================================================================

namespace std {
#if MQ_HAS_CONCEPTS
template <typename T>
    requires(!std::same_as<std::remove_cvref_t<T>, CircuitManager::qubit_t>)
bool operator==(const std::vector<CircuitManager::qubit_t>& lhs, const std::vector<T>& rhs) {
    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}
#else
template <typename T, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, CircuitManager::qubit_t>>>
bool operator==(const std::vector<CircuitManager::qubit_t>& lhs, const std::vector<T>& rhs) {
    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}
#endif  // MQ_HAS_CONCEPTS
}  // namespace std

// =============================================================================

namespace {
using qubit_t = CircuitManager::qubit_t;

template <typename T>
struct conv_helper {
    static constexpr auto value(T t) {
        return qubit_t{t};
    }
};

template <>
struct conv_helper<qubit_t> {
    static constexpr auto value(qubit_t qubit) {
        return qubit;
    }
};

template <typename... idx_t>
constexpr auto make_qubits(idx_t&&... idx) {
    return std::vector<qubit_t>{conv_helper<idx_t>::value(std::forward<idx_t>(idx))...};
}
}  // namespace

// =============================================================================

TEST_CASE("PhysicalView/No mapping", "[projectq_view][core]") {
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

    manager.apply_operator(op1_t(), {}, {qubit0});
    manager.apply_operator(op2_t(), {qubit1}, {qubit0});
    manager.apply_operator(op3_t(), {qubit0}, {qubit1});

    std::vector<instruction_t> instructions;

    SECTION("foreach_instructions (no mapping)") {
        manager.commit_changes();
        manager.apply_operator(tweedledum::Op::T(), {}, {qubit1});

        REQUIRE(!get::blocks(manager)[0].has_mapping());
        REQUIRE(std::size(get::blocks(manager)) == 2);

        auto view(manager.as_physical(committed));

        view.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 3);
        CHECK(instructions[0].kind() == op1_t::kind());
        CHECK(instructions[0].qubits() == qubit_array_t{qubit_t(qubit0)});
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == qubit_array_t{qubit_t(qubit1), qubit_t(qubit0)});
        CHECK(instructions[2].kind() == op3_t::kind());
        CHECK(instructions[2].qubits() == qubit_array_t{qubit_t(qubit0), qubit_t(qubit1)});
    }

    SECTION("foreach_r_instructions (no mapping)") {
        manager.commit_changes();
        manager.apply_operator(tweedledum::Op::T(), {}, {qubit1});

        REQUIRE(!get::blocks(manager)[0].has_mapping());
        REQUIRE(std::size(get::blocks(manager)) == 2);

        auto view(manager.as_physical(committed));

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

// -----------------------------------------------------------------------------

TEST_CASE("PhysicalView/With mapping", "[projectq_view][core]") {
    using circuit_t = CircuitManager::circuit_t;
    using instruction_t = CircuitManager::instruction_t;
    using qubit_t = CircuitManager::qubit_t;
    using qubit_array_t = std::vector<qubit_t>;
    using device_t = CircuitManager::device_t;
    using placement_t = CircuitManager::placement_t;
    using mapping_t = CircuitManager::mapping_t;

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

    manager.apply_operator(op1_t(), {}, {qubit0});
    manager.apply_operator(op2_t(), {qubit1}, {qubit0});
    manager.apply_operator(op3_t(), {qubit0}, {qubit1});

    auto& block = get::blocks(manager)[0];
    REQUIRE(!block.has_mapping());

    const auto device = tweedledum::Device::path(3);
    manager.apply_mapping(
        device,
        // Cold start
        [](const device_t& device, const circuit_t& circuit) {
            auto placement = *tweedledum::trivial_place(device, circuit);
            placement.swap_qubits(qubit_t(1), qubit_t(2));
            placement.swap_qubits(qubit_t(0), qubit_t(1));

            mapping_t mapping(placement);

            auto mapped = shallow_duplicate(circuit);

            circuit.foreach_instruction([&mapped, &placement = mapping.placement](const instruction_t& inst) {
                std::vector<qubit_t> new_wires;
                inst.foreach_qubit(
                    [&placement, &new_wires](const qubit_t& ref) { new_wires.emplace_back(placement.v_to_phy(ref)); });
                mapped.apply_operator(inst, new_wires);
            });
            return std::make_pair(mapped, mapping);
        },
        // Hot start
        [](const device_t& device, const circuit_t& circuit, placement_t& placement) {
            throw std::runtime_error("Error");
            return std::make_pair(circuit_t{}, mapping_t{placement});
        });

    const auto init_placement = get::initial_mapping(get::blocks(manager)[0]);

    REQUIRE(std::size(get::blocks(manager)) == 1);
    REQUIRE(block.has_mapping());
    REQUIRE(init_placement == get::final_mapping(block));
    REQUIRE(init_placement.v_to_phy() == make_qubits(1U, 2U, 0U));

    manager.commit_changes();
    manager.apply_operator(tweedledum::Op::T(), {}, {qubit1});

    REQUIRE(std::size(get::blocks(manager)) == 2);

    std::vector<instruction_t> instructions;

    SECTION("foreach_instructions (with mapping)") {
        auto view(manager.as_physical(committed));

        view.foreach_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 3);
        CHECK(instructions[0].kind() == op1_t::kind());
        CHECK(instructions[0].qubits() == make_qubits(init_placement.v_to_phy(0)));
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == make_qubits(init_placement.v_to_phy(1), init_placement.v_to_phy(0)));
        CHECK(instructions[2].kind() == op3_t::kind());
        CHECK(instructions[2].qubits() == make_qubits(init_placement.v_to_phy(0), init_placement.v_to_phy(1)));
    }

    SECTION("foreach_r_instructions (with mapping)") {
        auto view(manager.as_physical(committed));

        view.foreach_r_instruction([&instructions](const instruction_t& inst) { instructions.emplace_back(inst); });

        REQUIRE(std::size(instructions) == 3);
        CHECK(instructions[0].kind() == op3_t::kind());
        CHECK(instructions[0].qubits() == make_qubits(init_placement.v_to_phy(0), init_placement.v_to_phy(1)));
        CHECK(instructions[1].kind() == op2_t::kind());
        CHECK(instructions[1].qubits() == make_qubits(init_placement.v_to_phy(1), init_placement.v_to_phy(0)));
        CHECK(instructions[2].kind() == op1_t::kind());
        CHECK(instructions[2].qubits() == make_qubits(init_placement.v_to_phy(0)));
    }
}

// =============================================================================
