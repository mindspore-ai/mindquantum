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

#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/core/compute.hpp"
#include "experimental/core/control.hpp"
#include "experimental/decompositions/atom_meta.hpp"
#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/details/concepts.hpp"
#include "experimental/decompositions/details/decomposition_param.hpp"
#include "experimental/decompositions/gate_decomposer.hpp"
#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/non_gate_decomposition_rule.hpp"
#include "experimental/decompositions/trivial_atom.hpp"
#include "experimental/ops/gates.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

using mindquantum::catch2::Equals;
namespace ops = mindquantum::ops;
namespace decompositions = mindquantum::decompositions;

// =============================================================================

namespace {
class X2Z : public decompositions::GateDecompositionRule<X2Z, std::tuple<ops::X>, SINGLE_TGT_ANY_CTRL, ops::H, ops::Z> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "X2Z"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op*/,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0]});
        atom<ops::Z>()->apply(circuit, ops::Z{}, {qubits[0]});
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0]});
    }
};

class AllToX : public decompositions::NonGateDecompositionRule<AllToX, ops::X> {
 public:
    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "AllToX"sv;
    }

    bool is_applicable(const mindquantum::instruction_t& inst) const {
        return true;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::instruction_t& inst) {
        atom<ops::X>()->apply(circuit, ops::X{}, inst.qubits());
    }
};

class PseudoIdentityRule : public decompositions::NonGateDecompositionRule<PseudoIdentityRule> {
 public:
    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "PseudoIdentityRule"sv;
    }

    bool is_applicable(const mindquantum::instruction_t& inst) const {
        return true;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::instruction_t& inst) {
        if (auto* atom{storage().get_atom_for(inst)}; atom != nullptr) {
            atom->apply(circuit, inst);
        } else {
            circuit.apply_operator(inst);
        }
    }
};
}  // namespace

// -----------------------------------------------------------------------------

TEST_CASE("GateDecomposer/All2X", "[decompositions][atom]") {
    decompositions::GateDecomposer decomposer;
    REQUIRE(decomposer.num_atoms() == 0UL);
    REQUIRE(decomposer.num_rules() == 0UL);

    using circuit_t = mindquantum::circuit_t;
    using instruction_t = mindquantum::instruction_t;
    using h_t = decompositions::TrivialSimpleAtom<ops::H>;
    using ch_t = decompositions::TrivialSimpleAtom<ops::H, 1>;
    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using cx_t = decompositions::TrivialSimpleAtom<ops::X, 1>;

    auto* h_atom = decomposer.add_or_replace_atom<h_t>();
    CHECK(decomposer.num_atoms() == 1UL);
    CHECK(decomposer.num_rules() == 0UL);
    CHECK(h_atom != nullptr);

    CHECK(decomposer.has_atom<h_t>());
    CHECK(!decomposer.has_atom<ch_t>());
    CHECK(!decomposer.has_atom<x_t>());
    CHECK(!decomposer.has_atom<cx_t>());
    CHECK(!decomposer.has_atom<::AllToX>());

    auto* all2x_atom = decomposer.add_or_replace_atom<::AllToX>();
    CHECK(decomposer.num_atoms() == 2UL);  // H, X
    CHECK(decomposer.num_rules() == 1UL);
    CHECK(all2x_atom != nullptr);

    CHECK(decomposer.has_atom<h_t>());
    CHECK(!decomposer.has_atom<ch_t>());
    CHECK(decomposer.has_atom<x_t>());
    CHECK(!decomposer.has_atom<cx_t>());
    CHECK(decomposer.has_atom<::AllToX>());

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();
    auto inst_ref0 = original.apply_operator(ops::Y(), {q0});
    auto inst_ref1 = original.apply_operator(ops::H(), {q0, q1, q2});
    auto inst_ref2 = original.apply_operator(ops::Rx(1.0), {q2, q1, q0});

    auto inst0 = original.instruction(inst_ref0);
    auto inst1 = original.instruction(inst_ref1);
    auto inst2 = original.instruction(inst_ref2);

    CHECK(all2x_atom == decomposer.get_atom_for(inst0));
    CHECK(h_atom == decomposer.get_atom_for(inst1));
    CHECK(all2x_atom == decomposer.get_atom_for(inst2));

    auto reference = tweedledum::shallow_duplicate(original);
    reference.apply_operator(ops::X(), {q0});
    reference.apply_operator(ops::X(), {q0, q1, q2});
    reference.apply_operator(ops::X(), {q2, q1, q0});

    auto decomposed = tweedledum::shallow_duplicate(original);
    original.foreach_instruction(
        [&decomposed, &all2x_atom](const instruction_t& inst) { all2x_atom->apply(decomposed, inst); });

    CHECK_THAT(decomposed, Equals(reference));
}

TEST_CASE("GateDecomposer/Identity", "[decompositions][atom]") {
    decompositions::GateDecomposer decomposer;
    REQUIRE(decomposer.num_atoms() == 0UL);
    REQUIRE(decomposer.num_rules() == 0UL);

    using circuit_t = mindquantum::circuit_t;
    using instruction_t = mindquantum::instruction_t;

    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using h_t = decompositions::TrivialSimpleAtom<ops::H>;
    using z_t = decompositions::TrivialSimpleAtom<ops::Z>;
    using x2z_t = ::X2Z;
    using id_t = ::PseudoIdentityRule;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();
    original.apply_operator(ops::Y(), {q0});
    original.apply_operator(ops::X(), {q1, q2, q0});
    original.apply_operator(ops::Rx(1.0), {q2, q1, q0});

    SECTION("Identity") {
        auto* id_atom = decomposer.add_or_replace_atom<id_t>();
        CHECK(decomposer.num_atoms() == 0UL);
        CHECK(decomposer.num_rules() == 1UL);
        CHECK(id_atom != nullptr);

        CHECK(decomposer.has_atom<id_t>());
        CHECK(!decomposer.has_atom<x_t>());
        CHECK(!decomposer.has_atom<h_t>());
        CHECK(!decomposer.has_atom<z_t>());
        CHECK(!decomposer.has_atom<x2z_t>());

        auto decomposed = tweedledum::shallow_duplicate(original);
        original.foreach_instruction(
            [&decomposed, &id_atom](const instruction_t& inst) { id_atom->apply(decomposed, inst); });

        CHECK_THAT(decomposed, Equals(original));
    }

    SECTION("Identity + X2Z") {
        auto reference = tweedledum::shallow_duplicate(original);
        reference.apply_operator(ops::Y(), {q0});
        reference.apply_operator(ops::H(), {q1, q2, q0});
        reference.apply_operator(ops::Z(), {q1, q2, q0});
        reference.apply_operator(ops::H(), {q1, q2, q0});
        reference.apply_operator(ops::Rx(1.0), {q2, q1, q0});

        auto* id_atom = decomposer.add_or_replace_atom<id_t>();
        auto* x2z_atom = decomposer.add_or_replace_atom<x2z_t>();

        CHECK(decomposer.num_atoms() == 3UL);
        CHECK(decomposer.num_rules() == 1UL);
        CHECK(id_atom != nullptr);

        CHECK(decomposer.has_atom<id_t>());
        CHECK(!decomposer.has_atom<x_t>());
        CHECK(decomposer.has_atom<h_t>());
        CHECK(decomposer.has_atom<z_t>());
        CHECK(decomposer.has_atom<x2z_t>());

        auto decomposed = tweedledum::shallow_duplicate(original);
        original.foreach_instruction(
            [&decomposed, &id_atom](const instruction_t& inst) { id_atom->apply(decomposed, inst); });

        CHECK_THAT(decomposed, Equals(reference));
    }
}
