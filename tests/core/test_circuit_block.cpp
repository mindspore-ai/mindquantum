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

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>

#include <tweedledum/Operators/Standard.h>
#include <tweedledum/Passes/Mapping/Placer/TrivialPlacer.h>
#include <tweedledum/Passes/Mapping/jit_map.h>
#include <tweedledum/Passes/Mapping/sabre_map.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>
#include <tweedledum/Target/Device.h>
#include <tweedledum/Target/Placement.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/core/circuit_block.hpp"
#include "experimental/ops/gates/measure.hpp"

#include <catch2/catch_test_macros.hpp>

// -----------------------------------------------------------------------------

using namespace mindquantum::catch2;
namespace td = tweedledum;

// =============================================================================

class UnitTestAccessor {
 public:
    using block_t = mindquantum::CircuitBlock;

    static auto size(const block_t& block) {
        return std::size(block);
    }

    static auto& device(block_t& block) {
        return block.device_;
    }

    static auto& circuit(block_t& block) {
        return block.circuit_;
    }

    static auto& ext_to_td(block_t& block) {
        return block.ext_to_td_;
    }

    static auto& td_to_pq(block_t& block) {
        return block.qtd_to_ext_;
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
using mindquantum::CircuitBlock;

static auto not_equal_initial_final_mappings(get::block_t& block) {
#if MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
    return get::initial_mapping(block) != get::final_mapping(block);
#else
    return !(get::initial_mapping(block) == get::final_mapping(block));
#endif  // MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
}

// =============================================================================

namespace {
using qubit_t = CircuitBlock::qubit_t;

template <typename T>
struct conv_helper {
    static constexpr auto value(T t) {
        return qubit_t{static_cast<uint32_t>(t)};
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

TEST_CASE("CircuitBlock/Construction default", "[circuit_block][core]") {
    mindquantum::CircuitBlock block;
    CHECK(get::device(block) == nullptr);
    CHECK(std::size(get::circuit(block)) == 0);
    CHECK(get::circuit(block).num_wires() == 0);
}

TEST_CASE("CircuitBlock/Add qubits", "[circuit_block][core]") {
    using qubit_t = td::Qubit;
    using qubit_id_t = mindquantum::qubit_id_t;
    mindquantum::CircuitBlock block;

    SECTION("Add qubit") {
        const auto qubit0 = mindquantum::QubitID{10};
        REQUIRE(block.add_qubit(qubit0));
        REQUIRE(!block.add_qubit(qubit0));

        const auto& ext_to_td = get::ext_to_td(block);
        CHECK(std::size(ext_to_td) == 1);
        CHECK(ext_to_td.count(qubit0) == 1);

        const auto& td_to_pq = get::td_to_pq(block);
        CHECK(std::size(td_to_pq) == 1);
        CHECK(std::begin(td_to_pq)->second == qubit0);

        {
            const auto& ext_ids = block.ext_ids();
            CHECK(ext_ids == decltype(ext_ids){qubit0});

            const auto& td_ids = block.td_ids();
            CHECK(td_ids == decltype(td_ids){td::Qubit(0)});
        }

        CHECK(get::device(block) == nullptr);
        CHECK(std::size(get::circuit(block)) == 0);
        CHECK(get::circuit(block).num_wires() == 1);
        CHECK(!get::mapping(block));

        // ------------------------------------

        const auto qubit1 = mindquantum::QubitID{11};
        REQUIRE(block.add_qubit(qubit1));

        CHECK(std::size(ext_to_td) == 2);
        CHECK(ext_to_td.count(qubit0) == 1);
        CHECK(ext_to_td.count(qubit1) == 1);

        CHECK(std::size(td_to_pq) == 2);
        CHECK(std::all_of(std::begin(td_to_pq), std::end(td_to_pq),
                          [qubit0, qubit1](const auto& el) { return el.second == qubit0 || el.second == qubit1; }));

        {
            const auto& ext_ids = block.ext_ids();
            CHECK(ext_ids == decltype(ext_ids){qubit0, qubit1});

            const auto& td_ids = block.td_ids();
            CHECK(td_ids == decltype(td_ids){qubit_t(0), qubit_t(1)});
        }

        CHECK(get::device(block) == nullptr);
        CHECK(std::size(get::circuit(block)) == 0);
        CHECK(get::circuit(block).num_wires() == 2);
        CHECK(!get::mapping(block));
    }

    SECTION("Add qubit with device") {
        const auto device = tweedledum::Device::path(3);
        get::device(block) = &device;

        REQUIRE(block.add_qubit(mindquantum::QubitID{10}));
        REQUIRE(block.add_qubit(mindquantum::QubitID{11}));
        REQUIRE(block.add_qubit(mindquantum::QubitID{12}));
        CHECK(!block.add_qubit(mindquantum::QubitID{13}));
        CHECK(!block.has_qubit(mindquantum::QubitID{13}));
        CHECK(!block.add_qubit(mindquantum::QubitID{14}));
        CHECK(!block.has_qubit(mindquantum::QubitID{14}));
    }

    SECTION("Has qubit") {
        const auto qubit_id = 10;
        REQUIRE(block.add_qubit(mindquantum::QubitID{qubit_id}));
        CHECK(block.has_qubit(mindquantum::QubitID{qubit_id}));
        CHECK(!block.has_qubit(mindquantum::QubitID{qubit_id + 1}));
    }

    SECTION("Translate IDs") {
        using mindquantum::QubitID;
        const auto qubit0 = QubitID(10);
        const auto qubit1 = QubitID(11);

        REQUIRE(block.add_qubit(qubit0));
        REQUIRE(block.add_qubit(qubit1));

        CHECK(block.translate_id(qubit0) == qubit_t(0));
        CHECK(block.translate_id(qubit1) == qubit_t(1));

        CHECK(block.translate_id(qubit_t(0)) == qubit_id_t{qubit0});
        CHECK(block.translate_id(qubit_t(1)) == qubit_id_t{qubit1});
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/ID conversion", "[circuit_block][core]") {
    mindquantum::CircuitBlock block;

    const auto qubit0(mindquantum::QubitID{10});
    const auto qubit1(mindquantum::QubitID{200});
    const auto qubit2(mindquantum::QubitID{5});

    REQUIRE(block.add_qubit(qubit0));
    REQUIRE(block.add_qubit(qubit1));
    REQUIRE(block.add_qubit(qubit2));

    REQUIRE(block.has_qubit(qubit0));
    REQUIRE(block.has_qubit(qubit1));
    REQUIRE(block.has_qubit(qubit2));

    SECTION("External IDs") {
        const auto ext_ids = block.ext_ids();
        using v_t = decltype(ext_ids)::value_type;
        std::set<v_t> ref{qubit0, qubit1, qubit2};
        REQUIRE(std::set<v_t>(std::begin(ext_ids), std::end(ext_ids)) == ref);
    }

    SECTION("Tweedledum IDs") {
        const auto td_ids = block.td_ids();
        using v_t = decltype(td_ids)::value_type;
        using qubit_t = tweedledum::Qubit;

        std::set<v_t> ref{qubit_t(0), qubit_t(1), qubit_t(2)};
        REQUIRE(std::set<v_t>(std::begin(td_ids), std::end(td_ids)) == ref);
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/Translate IDs", "[circuit_block][core]") {
    mindquantum::CircuitBlock block;

    const auto qubit0(0), qubit1(10), qubit2(125), qubit3(3541);

    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit2}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit3}));

    const auto ext_ids = block.ext_ids();
    const auto td_ids = block.td_ids();

    SECTION("TD -> PQ") {
        std::set<unsigned int> qubit_ids;
        for (const auto& qubit_id : td_ids) {
            qubit_ids.emplace(block.translate_id(qubit_id));
        }

        CHECK(qubit_ids == std::set<unsigned int>{qubit0, qubit1, qubit2, qubit3});
    }

    SECTION("TD -> PQ (td::Qubit)") {
        std::set<unsigned int> qubit_ids;
        for (const auto& qubit_id : td_ids) {
            qubit_ids.emplace(block.translate_id_td(qubit_id));
        }

        CHECK(qubit_ids == std::set<unsigned int>{qubit0, qubit1, qubit2, qubit3});
    }

    SECTION("PQ -> TD") {
        std::set<unsigned int> qubit_ids;
        for (const auto& qubit_id : ext_ids) {
            qubit_ids.emplace(static_cast<unsigned int>(block.translate_id(qubit_id)));
        }

        CHECK(qubit_ids == std::set<unsigned int>{0, 1, 2, 3});
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/Chaining constructors", "[circuit_block][core]") {
    using qubit_t = CircuitBlock::qubit_t;
    using device_t = CircuitBlock::device_t;
    using placement_t = tweedledum::Placement;
    using mapping_t = CircuitBlock::mapping_t;

    mindquantum::CircuitBlock parent;

    const auto qubit0(mindquantum::QubitID{10});
    const auto qubit1(mindquantum::QubitID{200});
    const auto qubit2(mindquantum::QubitID{5});

    REQUIRE(parent.add_qubit(qubit0));
    REQUIRE(parent.add_qubit(qubit1));
    REQUIRE(parent.add_qubit(qubit2));
    REQUIRE(!parent.has_mapping());
    REQUIRE(std::size(parent) == 0);

    SECTION("Chaining no mapping") {
        mindquantum::CircuitBlock child(parent, mindquantum::CircuitBlock::chain_ctor);
        CHECK(!child.has_mapping());
        CHECK(std::size(parent) == 0);
        CHECK(std::size(child) == 0);

        std::vector<qubit_t> parent_qubits, child_qubits;
        get::circuit(parent).foreach_qubit(
            [&parent_qubits](const qubit_t& qubit) { parent_qubits.emplace_back(qubit); });
        get::circuit(child).foreach_qubit([&child_qubits](const qubit_t& qubit) { child_qubits.emplace_back(qubit); });

        CHECK(parent_qubits == child_qubits);
    }

    SECTION("Chaining with mapping") {
        const auto device = device_t::path(3);
        get::device(parent) = &device;

        placement_t init_placement(3, 3);
        init_placement.map_v_phy(qubit_t{0}, qubit_t{2});
        init_placement.map_v_phy(qubit_t{1}, qubit_t{0});
        init_placement.map_v_phy(qubit_t{2}, qubit_t{1});

        mapping_t mapping(init_placement);
        mapping.placement.map_v_phy(qubit_t{0}, qubit_t{1});
        mapping.placement.map_v_phy(qubit_t{1}, qubit_t{2});
        mapping.placement.map_v_phy(qubit_t{2}, qubit_t{0});
        get::mapping(parent).emplace(mapping);
        CHECK(not_equal_initial_final_mappings(parent));

        mindquantum::CircuitBlock child(parent, mindquantum::CircuitBlock::chain_ctor);
        CHECK(parent.has_mapping());
        CHECK(child.has_mapping());
        CHECK(std::size(child) == 0);

        std::vector<qubit_t> parent_qubits, child_qubits;
        get::circuit(parent).foreach_qubit(
            [&parent_qubits](const qubit_t& qubit) { parent_qubits.emplace_back(qubit); });
        get::circuit(child).foreach_qubit([&child_qubits](const qubit_t& qubit) { child_qubits.emplace_back(qubit); });

        CHECK(parent_qubits == child_qubits);
        CHECK(not_equal_initial_final_mappings(parent));
        CHECK(get::final_mapping(parent) == get::initial_mapping(child));
    }

    SECTION("Chaining no mapping delete qubits") {
        mindquantum::CircuitBlock child(parent, {qubit0, qubit2}, mindquantum::CircuitBlock::chain_ctor);
        CHECK(!child.has_mapping());
        CHECK(std::size(child) == 0);

        std::vector<qubit_t> parent_qubits, child_qubits;
        get::circuit(parent).foreach_qubit(
            [&parent_qubits](const qubit_t& qubit) { parent_qubits.emplace_back(qubit); });
        get::circuit(child).foreach_qubit([&child_qubits](const qubit_t& qubit) { child_qubits.emplace_back(qubit); });

        CHECK(std::size(parent_qubits) == 3);
        CHECK(std::size(child_qubits) == 1);

        CHECK(!child.has_qubit(qubit0));
        CHECK(child.has_qubit(qubit1));
        CHECK(!child.has_qubit(qubit2));
    }

    SECTION("Chaining with mapping delete qubits") {
        using mindquantum::QubitID;

        const auto device = device_t::path(3);
        get::device(parent) = &device;

        placement_t init_placement(3, 3);
        init_placement.map_v_phy(qubit_t{0}, qubit_t{2});
        init_placement.map_v_phy(qubit_t{1}, qubit_t{0});
        init_placement.map_v_phy(qubit_t{2}, qubit_t{1});

        mapping_t mapping(init_placement);
        mapping.placement.map_v_phy(qubit_t{0}, qubit_t{1});
        mapping.placement.map_v_phy(qubit_t{1}, qubit_t{2});
        mapping.placement.map_v_phy(qubit_t{2}, qubit_t{0});
        get::mapping(parent) = mapping;
        CHECK(not_equal_initial_final_mappings(parent));

        mindquantum::CircuitBlock child(parent, {qubit0, qubit2}, mindquantum::CircuitBlock::chain_ctor);
        CHECK(parent.has_mapping());
        CHECK(child.has_mapping());
        CHECK(std::size(child) == 0);

        std::vector<qubit_t> parent_qubits, child_qubits;
        get::circuit(parent).foreach_qubit(
            [&parent_qubits](const qubit_t& qubit) { parent_qubits.emplace_back(qubit); });
        get::circuit(child).foreach_qubit([&child_qubits](const qubit_t& qubit) { child_qubits.emplace_back(qubit); });

        CHECK(std::size(parent_qubits) == 3);
        CHECK(std::size(child_qubits) == 1);

        CHECK(!child.has_qubit(qubit0));
        CHECK(child.has_qubit(qubit1));
        CHECK(!child.has_qubit(qubit2));

        CHECK(not_equal_initial_final_mappings(parent));
        CHECK(get::initial_mapping(child).v_to_phy() == make_qubits(2));
        CHECK(get::initial_mapping(child).phy_to_v() == make_qubits(qubit_t::invalid(), qubit_t::invalid(), 0));

        CHECK(get::initial_mapping(child).v_to_phy(qubit_t{0})
              == get::final_mapping(parent).v_to_phy(parent.translate_id(QubitID(qubit1))));
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/Apply operators", "[circuit_block][core]") {
    using cbit_t = CircuitBlock::cbit_t;
    using qubit_t = CircuitBlock::qubit_t;
    using instruction_t = CircuitBlock::instruction_t;

    mindquantum::CircuitBlock block;
    const auto qubit0(10);
    const auto qubit1(200);
    const auto qubit2(5);

    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit2}));
    REQUIRE(std::size(block) == 0);

    SECTION("General method") {
        block.apply_operator(tweedledum::Op::H(), {}, {qubit0});
        block.apply_operator(tweedledum::Op::X(), {qubit0}, {qubit1});
        CHECK(std::size(block) == 2);

        std::vector<const instruction_t*> instructions;
        get::circuit(block).foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });

        CHECK(std::size(instructions) == 2);
        CHECK(instructions[0]->kind() == tweedledum::Op::H::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[1]->kind() == tweedledum::Op::X::kind());
        CHECK(instructions[1]->qubits() == std::vector<qubit_t>{qubit_t(0), qubit_t(1)});
    }

    SECTION("Tweedledum instruction overload") {
        tweedledum::Circuit circuit;
        const auto q0 = circuit.create_qubit();
        const auto q1 = circuit.create_qubit();
        const auto inst_ref = circuit.apply_operator(tweedledum::Op::X(), {q0, q1});
        block.apply_operator(circuit.instruction(inst_ref), {}, {qubit0});
        block.apply_operator(circuit.instruction(inst_ref), {qubit0}, {qubit1});
        CHECK(std::size(block) == 2);

        std::vector<const instruction_t*> instructions;
        get::circuit(block).foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });
        CHECK(std::size(instructions) == 2);
        CHECK(instructions[0]->kind() == tweedledum::Op::X::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[1]->kind() == tweedledum::Op::X::kind());
        CHECK(instructions[1]->qubits() == std::vector<qubit_t>{qubit_t(0), qubit_t(1)});
    }

    SECTION("Measurement") {
        block.apply_measurement(qubit0);
        CHECK(std::size(block) == 1);
        CHECK(get::circuit(block).num_wires() == 4);  // 3 qubits + 1 cbit

        CHECK(block.has_cbit(mindquantum::QubitID{qubit0}));

        std::vector<const instruction_t*> instructions;
        get::circuit(block).foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });
        CHECK(std::size(instructions) == 1);
        CHECK(instructions[0]->kind() == mindquantum::ops::Measure::kind());
        CHECK(instructions[0]->qubits() == std::vector<qubit_t>{qubit_t(0)});
        CHECK(instructions[0]->cbits() == std::vector<cbit_t>{cbit_t(0)});
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/Mapping", "[circuit_block][core]") {
    using instruction_t = CircuitBlock::instruction_t;
    using qubit_t = CircuitBlock::qubit_t;
    using circuit_t = CircuitBlock::circuit_t;
    using device_t = CircuitBlock::device_t;
    using placement_t = CircuitBlock::placement_t;
    using mapping_t = CircuitBlock::mapping_t;
    namespace Op = tweedledum::Op;

    mindquantum::CircuitBlock block;

    const auto get_td_id = [&](mindquantum::qubit_id_t ext_id) {
        return std::get<0>(get::ext_to_td(block).at(mindquantum::QubitID{ext_id}));
    };

    const auto qubit0(10);
    const auto qubit1(11);
    const auto qubit2(12);
    const auto qubit3(13);
    const auto qubit4(14);
    const auto qubit5(15);
    const auto qubit6(16);

    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit2}));

    SECTION("Simple 3-qubits") {
        const auto device = tweedledum::Device::path(3);
        block.apply_operator(tweedledum::Op::X(), {qubit0}, {qubit1});
        block.apply_operator(tweedledum::Op::Y(), {qubit1}, {qubit2});
        block.apply_operator(tweedledum::Op::Z(), {qubit0}, {qubit2});
        REQUIRE(std::size(block) == 3);

        CHECK(get::device(block) == nullptr);

        CHECK(get_td_id(qubit0) == qubit_t(0));
        CHECK(get_td_id(qubit1) == qubit_t(1));
        CHECK(get_td_id(qubit2) == qubit_t(2));

        block.apply_mapping(
            device,
            // Cold start
            [](const device_t& device, const circuit_t& circuit) {
                auto placement = *tweedledum::trivial_place(device, circuit);
                sabre_re_place(device, circuit, placement);
                tweedledum::SabreRouter router(device, circuit, placement);
                return router.run();
            },
            // Hot start
            [](const device_t& device, const circuit_t& circuit, placement_t& placement) {
                tweedledum::SabreRouter router(device, circuit, placement);
                return router.run();
            });

        CHECK(get::device(block) == &device);

        CHECK(get_td_id(qubit0) == qubit_t(1));
        CHECK(get_td_id(qubit1) == qubit_t(0));
        CHECK(get_td_id(qubit2) == qubit_t(2));

        circuit_t ref;
        auto q0 = ref.create_qubit();
        auto q1 = ref.create_qubit();
        auto q2 = ref.create_qubit();
        ref.apply_operator(Op::X(), {q0, q1});
        ref.apply_operator(Op::Y(), {q1, q2});
        ref.apply_operator(Op::Swap(), {q0, q1});
        ref.apply_operator(Op::Z(), {q1, q2});

        CHECK_THAT(get::circuit(block), Equals(ref));

        CHECK(get::initial_mapping(block).v_to_phy() == make_qubits(0, 1, 2));
        CHECK(get::initial_mapping(block).phy_to_v() == make_qubits(0, 1, 2));
        CHECK(get::final_mapping(block).v_to_phy() == make_qubits(1, 0, 2));
        CHECK(get::final_mapping(block).phy_to_v() == make_qubits(1, 0, 2));
    }
}

// =============================================================================

TEST_CASE("CircuitBlock/Transform", "[circuit_block][core]") {
    using circuit_t = CircuitBlock::circuit_t;
    using instruction_t = CircuitBlock::instruction_t;
    namespace Op = tweedledum::Op;

    mindquantum::CircuitBlock block;
    const auto qubit0(10);
    const auto qubit1(200);
    const auto qubit2(5);

    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit0}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit1}));
    REQUIRE(block.add_qubit(mindquantum::QubitID{qubit2}));

    block.apply_operator(tweedledum::Op::X(), {qubit0}, {qubit1});
    block.apply_operator(tweedledum::Op::X(), {qubit1}, {qubit2});
    block.apply_operator(tweedledum::Op::X(), {qubit0}, {qubit2});
    REQUIRE(get::circuit(block).num_qubits() == 3);
    REQUIRE(std::size(block) == 3);

    SECTION("Transform to empty circuit") {
        block.transform([](const circuit_t& circuit) { return circuit_t{}; });

        circuit_t ref;
        CHECK_THAT(get::circuit(block), Equals(ref));
    }

    SECTION("No change transform") {
        std::vector<const instruction_t*> instructions_ref;
        get::circuit(block).foreach_instruction(
            [&instructions_ref](const instruction_t& inst) { instructions_ref.emplace_back(&inst); });

        block.transform([](const circuit_t& circuit) {});

        std::vector<const instruction_t*> instructions;
        get::circuit(block).foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });

        CHECK(get::circuit(block).num_qubits() == 3);
        CHECK(std::size(block) == 3);
        CHECK_THAT(instructions, Equals(instructions_ref));
    }

    SECTION("Modifying transform") {
        std::vector<const instruction_t*> instructions_ref;
        get::circuit(block).foreach_instruction(
            [&instructions_ref](const instruction_t& inst) { instructions_ref.emplace_back(&inst); });

        block.transform([](const circuit_t& circuit) {
            auto new_circuit(tweedledum::shallow_duplicate(circuit));

            circuit.foreach_instruction([&new_circuit](const instruction_t& inst) {
                new_circuit.apply_operator(tweedledum::Op::H(), inst.qubits());
            });

            return new_circuit;
        });

        std::vector<const instruction_t*> instructions;
        get::circuit(block).foreach_instruction(
            [&instructions](const instruction_t& inst) { instructions.emplace_back(&inst); });

        circuit_t ref;
        auto q0 = ref.create_qubit();
        auto q1 = ref.create_qubit();
        auto q2 = ref.create_qubit();
        ref.apply_operator(Op::H(), {q0, q1});
        ref.apply_operator(Op::H(), {q1, q2});
        ref.apply_operator(Op::H(), {q0, q2});

        CHECK_THAT(get::circuit(block), Equals(ref));
    }
}

// =============================================================================
