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

#include "experimental/decompositions/atom_meta.hpp"
#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/trivial_atom.hpp"
#include "experimental/ops/gates.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

namespace ops = mindquantum::ops;
namespace decompositions = mindquantum::decompositions;

// ==============================================================================

class UnitTestAccessor {
 public:
    using decomposer_t = mindquantum::decompositions::AtomStorage;

    static void print(const decomposer_t& storage) {
        for (const auto& [key, val] : storage.atoms_) {
            const auto& [s, i] = key;
            std::cout << s << "," << i << ": " << &val << '\n';
        }
    }
};

// ==============================================================================

TEST_CASE("AtomStorage/trivial_atom", "[decompositions][atom]") {
    decompositions::AtomStorage storage;
    REQUIRE(std::size(storage) == 0UL);

    using h_t = decompositions::TrivialSimpleAtom<ops::H>;
    using ch_t = decompositions::TrivialSimpleAtom<ops::H, 1>;
    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using cx_t = decompositions::TrivialSimpleAtom<ops::X, 1>;

    SECTION("Single insertion") {
        const auto* atom = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);
        CHECK(atom != nullptr);

        CHECK(storage.has_atom<h_t>());
        CHECK(!storage.has_atom<ch_t>());
        CHECK(!storage.has_atom<x_t>());
        CHECK(!storage.has_atom<cx_t>());
    }

    SECTION("Same kind, same controls") {
        const auto* atom1 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);

        const auto* atom2 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);
        CHECK(atom1 == atom2);

        CHECK(storage.has_atom<h_t>());
        CHECK(!storage.has_atom<ch_t>());
        CHECK(!storage.has_atom<x_t>());
        CHECK(!storage.has_atom<cx_t>());
    }

    SECTION("Same kind, different controls") {
        const auto* atom1 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);

        const auto* atom2 = storage.add_or_replace_atom<ch_t>();
        CHECK(std::size(storage) == 2UL);
        CHECK(atom1 != atom2);

        CHECK(storage.has_atom<h_t>());
        CHECK(storage.has_atom<ch_t>());
        CHECK(!storage.has_atom<x_t>());
        CHECK(!storage.has_atom<cx_t>());
    }

    SECTION("Same kind, different controls 2") {
        const auto* atom1 = storage.add_or_replace_atom<ch_t>();
        CHECK(std::size(storage) == 1UL);

        const auto* atom2 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 2UL);
        CHECK(atom1 != atom2);

        CHECK(storage.has_atom<h_t>());
        CHECK(storage.has_atom<ch_t>());
        CHECK(!storage.has_atom<x_t>());
        CHECK(!storage.has_atom<cx_t>());
    }

    SECTION("Different kind, different controls 1") {
        const auto* atom1 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);

        const auto* atom2 = storage.add_or_replace_atom<x_t>();
        CHECK(std::size(storage) == 2UL);
        CHECK(atom1 != atom2);

        CHECK(storage.has_atom<h_t>());
        CHECK(!storage.has_atom<ch_t>());
        CHECK(storage.has_atom<x_t>());
        CHECK(!storage.has_atom<cx_t>());
    }

    SECTION("Different kind, different controls 2") {
        const auto* atom1 = storage.add_or_replace_atom<h_t>();
        CHECK(std::size(storage) == 1UL);

        const auto* atom2 = storage.add_or_replace_atom<cx_t>();
        CHECK(std::size(storage) == 2UL);
        CHECK(atom1 != atom2);

        CHECK(storage.has_atom<h_t>());
        CHECK(!storage.has_atom<ch_t>());
        CHECK(!storage.has_atom<x_t>());
        CHECK(storage.has_atom<cx_t>());
    }
}

// =============================================================================

namespace {
namespace atoms = decompositions::atoms;

class X2Z : public decompositions::GateDecompositionRule<X2Z, std::tuple<ops::X>, SINGLE_TGT_ANY_CTRL, ops::H, ops::Z> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "X2Z"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op */,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        // TODO(dnguyen): I find this syntax redundant, we should find a way to fix this!
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0]});
        atom<ops::Z>()->apply(circuit, ops::Z{}, {qubits[0]});
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0]});
    }
};

class Z2YDummy
    : public decompositions::GateDecompositionRule<Z2YDummy, std::tuple<ops::Z>, SINGLE_TGT_ANY_CTRL, ops::Y> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Z2YDummy"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op*/,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        atom<ops::Y>()->apply(circuit, ops::Y{}, qubits);
    }
};

class CNOT2CZDummy
    : public decompositions::GateDecompositionRule<CNOT2CZDummy, std::tuple<ops::X>, SINGLE_TGT_SINGLE_CTRL, ops::H,
                                                   ops::Z> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "CNOT2CZDummy"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op*/,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0], qubits[1]});
        atom<ops::Z>()->apply(circuit, ops::Z{}, {qubits[0], qubits[1]});
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0], qubits[1]});
    }
};

class CNOT2CZ
    : public decompositions::GateDecompositionRule<CNOT2CZ, std::tuple<ops::X>, SINGLE_TGT_SINGLE_CTRL, ops::H,
                                                   atoms::C<ops::Z>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "CNOT2CZ"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op*/,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[1]});
        atom<atoms::C<ops::Z>>()->apply(circuit, ops::Z{}, {qubits[0], qubits[1]});
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[1]});
    }
};
}  // namespace

// -----------------------------------------------------------------------------

TEST_CASE("AtomStorage/decomposition_rule", "[decompositions][atom]") {
    decompositions::AtomStorage storage;
    REQUIRE(std::size(storage) == 0UL);

    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using h_t = decompositions::TrivialSimpleAtom<ops::H>;
    using z_t = decompositions::TrivialSimpleAtom<ops::Z>;
    using x2z_t = ::X2Z;
    using cnot2cz_t = ::CNOT2CZ;
    using cnot2cz_dummy_t = ::CNOT2CZDummy;

    SECTION("Same kind, same controls") {
        const auto* atom1 = storage.add_or_replace_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 != nullptr);
        CHECK(!storage.has_atom<x_t>());
        CHECK(storage.has_atom<x2z_t>());

        const auto* atom2 = storage.add_or_replace_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 == atom2);
        CHECK(!storage.has_atom<x_t>());
        CHECK(storage.has_atom<x2z_t>());
    }

    SECTION("Same kind, same control") {
        const auto* atom1 = storage.add_or_replace_atom<x_t>();
        CHECK(std::size(storage) == 1UL);  // X
        CHECK(atom1 != nullptr);
        CHECK(storage.has_atom<x_t>());
        CHECK(!storage.has_atom<x2z_t>());

        const auto* atom2 = storage.add_or_replace_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 == atom2);
        CHECK(!storage.has_atom<x_t>());
        CHECK(storage.has_atom<x2z_t>());

        atom1 = storage.add_or_replace_atom<x_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 != nullptr);
        CHECK(storage.has_atom<x_t>());
        CHECK(!storage.has_atom<x2z_t>());

        atom2 = storage.add_or_return_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 == atom2);
        CHECK(storage.has_atom<x_t>());
        CHECK(!storage.has_atom<x2z_t>());
    }

    SECTION("Same kind, different controls: replacement") {
        const auto* atom1 = storage.add_or_replace_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 != nullptr);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(!storage.has_atom<cnot2cz_t>());

        const auto* atom2 = storage.add_or_replace_atom<cnot2cz_t>();
        CHECK(std::size(storage) == 4UL);  // X, Z, H, CNOT
        CHECK(atom1 != atom2);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(storage.has_atom<cnot2cz_t>());
    }

    SECTION("Same kind, different controls: insert or return") {
        const auto* atom1 = storage.add_or_return_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 != nullptr);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(!storage.has_atom<cnot2cz_t>());

        const auto* atom2 = storage.add_or_return_atom<cnot2cz_t>();
        CHECK(std::size(storage) == 4UL);  // X, Z, H, CNOT
        CHECK(atom1 != atom2);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(storage.has_atom<cnot2cz_t>());
    }

    SECTION("Same kind, different controls: insert or compatible") {
        const auto* atom1 = storage.add_or_return_atom<x2z_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom1 != nullptr);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(!storage.has_atom<cnot2cz_t>());

        const auto* atom2 = storage.add_or_compatible_atom<cnot2cz_t>();
        CHECK(std::size(storage) == 3UL);  // X, Z, H
        CHECK(atom2 != nullptr);
        CHECK(storage.has_atom<x2z_t>());
        CHECK(!storage.has_atom<cnot2cz_t>());
    }

    SECTION("Different kind, same controls: insert or replace") {
        using z2y_t = ::Z2YDummy;
        const auto* x_atom = storage.add_or_replace_atom<x_t>();
        const auto* z2y_atom = storage.add_or_replace_atom<z2y_t>();
        CHECK(std::size(storage) == 3UL);  // X, Y, Z
        CHECK(x_atom != nullptr);
        CHECK(z2y_atom != nullptr);
        CHECK(storage.has_atom<x_t>());
        CHECK(!storage.has_atom<x2z_t>());
        CHECK(!storage.has_atom<h_t>());
        CHECK(!storage.has_atom<z_t>());
        CHECK(storage.has_atom<z2y_t>());

        const auto* x2z_atom = storage.add_or_replace_atom<x2z_t>();
        CHECK(std::size(storage) == 4UL);  // X, Y, Z, H
        CHECK(x2z_atom != nullptr);
        CHECK(!storage.has_atom<x_t>());  // Make sure we replaced the X atom
        CHECK(storage.has_atom<x2z_t>());
        CHECK(storage.has_atom<h_t>());
        CHECK(!storage.has_atom<z_t>());  // Make sure that we did not replace the Z atom
        CHECK(storage.has_atom<z2y_t>());
    }
}

// =============================================================================

TEST_CASE("AtomStorage/get_atom_for", "[decompositions][atom]") {
    using instruction_t = mindquantum::instruction_t;
    using qubit_t = mindquantum::qubit_t;
    using qubits_t = mindquantum::qubits_t;

    decompositions::AtomStorage storage;
    REQUIRE(std::size(storage) == 0UL);

    using h_t = decompositions::TrivialSimpleAtom<ops::H>;
    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using cx_t = decompositions::TrivialSimpleAtom<ops::X, 1>;
    using x2z_t = ::X2Z;
    using cnot2cz_t = ::CNOT2CZ;

    const auto q0 = qubit_t{0};
    const auto q1 = qubit_t{0};
    const instruction_t x_inst{ops::X{}, {q0}, {}};
    const instruction_t cx_inst{ops::X{}, {q0, q1}, {}};

    const auto* h_atom = storage.add_or_replace_atom<h_t>();
    const auto* x_atom = storage.add_or_replace_atom<x_t>();
    const auto* cx_atom = storage.add_or_replace_atom<cx_t>();
    CHECK(std::size(storage) == 3UL);  // H, X, CNOT

    CHECK(h_atom->is_kind(ops::H::kind()));
    CHECK(x_atom->is_kind(ops::X::kind()));
    CHECK(cx_atom->is_kind(ops::X::kind()));

    CHECK(!h_atom->is_applicable(x_inst));
    CHECK(!h_atom->is_applicable(cx_inst));
    CHECK(x_atom->is_applicable(x_inst));
    CHECK(x_atom->is_applicable(cx_inst));
    CHECK(!cx_atom->is_applicable(x_inst));
    CHECK(cx_atom->is_applicable(cx_inst));

    CHECK(x_atom == storage.get_atom_for(x_inst));
    CHECK(cx_atom == storage.get_atom_for(cx_inst));

    const auto* cx_atom2 = storage.add_or_replace_atom<cnot2cz_t>();

    CHECK(std::size(storage) == 4UL);  // H, X, CNOT, CZ
    CHECK(cx_atom2 == cx_atom);
    CHECK(cx_atom2->name() == cnot2cz_t::name());

    CHECK(!h_atom->is_applicable(x_inst));
    CHECK(!h_atom->is_applicable(cx_inst));
    CHECK(x_atom->is_applicable(x_inst));
    CHECK(x_atom->is_applicable(cx_inst));
    CHECK(!cx_atom->is_applicable(x_inst));
    CHECK(cx_atom->is_applicable(cx_inst));

    CHECK(x_atom == storage.get_atom_for(x_inst));
    CHECK(cx_atom == storage.get_atom_for(cx_inst));

    const auto* x_atom2 = storage.add_or_replace_atom<x2z_t>();

    CHECK(std::size(storage) == 5UL);  // H, X, Z, CNOT, CZ
    CHECK(x_atom2 == x_atom);
    CHECK(x_atom2->name() == x2z_t::name());

    CHECK(!h_atom->is_applicable(x_inst));
    CHECK(!h_atom->is_applicable(cx_inst));
    CHECK(x_atom->is_applicable(x_inst));
    CHECK(x_atom->is_applicable(cx_inst));
    CHECK(!cx_atom->is_applicable(x_inst));
    CHECK(cx_atom->is_applicable(cx_inst));

    CHECK(x_atom == storage.get_atom_for(x_inst));
    CHECK(cx_atom == storage.get_atom_for(cx_inst));
}

// =============================================================================
