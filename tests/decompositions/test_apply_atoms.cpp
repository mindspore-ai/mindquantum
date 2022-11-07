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

#include <complex>

#if __has_include("tweedledum/../../tests/check_unitary.h")
#    include "tweedledum/../../tests/check_unitary.h"
#else
#    include <tweedledum/test/check_unitary.h>
#endif
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/core/circuit_block.hpp"
#include "experimental/decompositions/atom_meta.hpp"
#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/details/decomposition_param.hpp"
#include "experimental/decompositions/details/traits.hpp"
#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/trivial_atom.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/parametric/register_gate_type.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

namespace ops = mindquantum::ops;
namespace decompositions = mindquantum::decompositions;

#if __has_include(<numbers>) && __cplusplus > 201703L
static constexpr auto PI_VAL = std::numbers::pi;
#else
static constexpr auto PI_VAL = 3.141592653589793;
#endif  // __has_include(<numbers>) && C++20
static constexpr auto PI_VAL_2 = PI_VAL / 2.;

// ==============================================================================

class UnitTestAccessor {
 public:
    using storage_t = mindquantum::decompositions::AtomStorage;

    static void print(const storage_t& storage) {
        for (const auto& [key, val] : storage.atoms_) {
            const auto& [s, i] = key;
            std::cout << s << "," << i << ": " << &val << '\n';
        }
    }
};

namespace {
namespace atoms = decompositions::atoms;

using mindquantum::operator_t;
using mindquantum::qubit_t;

template <typename... args_t>
auto make_ops_array(args_t&&... args) {
    return std::array<operator_t, sizeof...(args_t)>{std::forward<args_t>(args)...};
}

template <typename... args_t>
auto make_int_array(args_t&&... args) {
    return std::array<int, sizeof...(args_t)>{std::forward<args_t>(args)...};
}

template <typename... args_t>
auto make_idx_vector(args_t&&... args) {
    return std::vector<uint16_t>{std::forward<args_t>(args)...};
}

template <typename... idx_t>
auto make_qubits(idx_t&&... idx) {
    return std::vector<qubit_t>{qubit_t(std::forward<idx_t>(idx))...};
}
}  // namespace

// =============================================================================

namespace SymEngine {
auto operator==(const SymEngine::Basic& a, const SymEngine::Number& b) {
    return eq(a, b);
}
auto operator==(const SymEngine::Basic& a, const SymEngine::Basic& b) {
    return eq(a, b);
}
}  // namespace SymEngine

// =============================================================================

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

// -----------------------------------------------------------------------------

TEST_CASE("GateDecompositionRule/X2Z", "[decompositions][atom]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<::X2Z>();

    const auto replaced = make_ops_array(mindquantum::ops::H(), mindquantum::ops::Z(), mindquantum::ops::H());
    auto qubits_ref = make_qubits();
    auto inst_ref = inst_ref_t{0};

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        qubits_ref = make_qubits(0);
        inst_ref = original.apply_operator(ops::X(), {q0});
    }

    SECTION("Single control qubit") {
        qubits_ref = make_qubits(0, 1);
        inst_ref = original.apply_operator(ops::X(), {q0, q1});
    }

    SECTION("Double control qubit") {
        qubits_ref = make_qubits(2, 0, 1);
        inst_ref = original.apply_operator(ops::X(), {q2, q0, q1});
    }

    SECTION("Triple control qubit") {
        qubits_ref = make_qubits(3, 0, 2, 1);
        inst_ref = original.apply_operator(ops::X(), {q3, q0, q2, q1});
    }

    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    CHECK(inst.kind() == ops::X::kind());
    CHECK(inst.qubits() == qubits_ref);
    REQUIRE(atom->is_applicable(inst));
    atom->apply(decomposed, inst);

    REQUIRE(std::size(decomposed) == 3);
    for (auto i(0UL); i < std::size(replaced); ++i) {
        INFO("  checking instruction at index: " << i);
        const auto& new_inst = decomposed.instruction(inst_ref_t(i));
        CHECK(new_inst.kind() == replaced[i].kind());
        CHECK(new_inst.qubits() == qubits_ref);
    }
}

// =============================================================================

class CNOT2CZ
    : public decompositions::GateDecompositionRule<CNOT2CZ, std::tuple<ops::X>, SINGLE_TGT_SINGLE_CTRL, ops::H,
                                                   atoms::C<ops::Z>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto kind() noexcept {
        return ops::X::kind();
    }

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

// -----------------------------------------------------------------------------

TEST_CASE("GateDecompositionRule/CNOT2C", "[decompositions][atom]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<::CNOT2CZ>();

    const auto replaced = make_ops_array(mindquantum::ops::H(), mindquantum::ops::Z(), mindquantum::ops::H());
    auto do_smth = false;
    auto h_qubits_ref = make_qubits();
    auto z_qubits_ref = make_qubits();
    auto inst_ref = inst_ref_t{0};

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        z_qubits_ref = make_qubits(0);
        inst_ref = original.apply_operator(ops::X(), {q0});
    }

    SECTION("Single control qubit") {
        do_smth = true;

        h_qubits_ref = make_qubits(1);
        z_qubits_ref = make_qubits(0, 1);
        inst_ref = original.apply_operator(ops::X(), {q0, q1});
    }

    SECTION("Double control qubit") {
        do_smth = true;
        // C(CNOT) -> q2 is "free" control qubit
        h_qubits_ref = make_qubits(2, 1);
        z_qubits_ref = make_qubits(2, 0, 1);
        inst_ref = original.apply_operator(ops::X(), {q2, q0, q1});
    }

    SECTION("Triple control qubit") {
        do_smth = true;
        // C(C(CNOT)) -> q3, q0 are "free" control qubit
        h_qubits_ref = make_qubits(3, 0, 1);
        z_qubits_ref = make_qubits(3, 0, 2, 1);
        inst_ref = original.apply_operator(ops::X(), {q3, q0, q2, q1});
    }

    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    CHECK(inst.kind() == ops::X::kind());
    CHECK(inst.qubits() == z_qubits_ref);

    if (do_smth) {
        // NB: This check should be removed if we use == instead of <= for the number of control qubit test in
        //     is_applicable()
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);

        REQUIRE(std::size(decomposed) == 3);

        {  // H
            const auto& new_inst = decomposed.instruction(inst_ref_t(0));
            CHECK(new_inst.kind() == replaced[0].kind());
            CHECK(new_inst.qubits() == h_qubits_ref);
        }

        {  // Z
            const auto& new_inst = decomposed.instruction(inst_ref_t(1));
            CHECK(new_inst.kind() == replaced[1].kind());
            CHECK(new_inst.qubits() == z_qubits_ref);
        }

        {  // H
            const auto& new_inst = decomposed.instruction(inst_ref_t(2));
            CHECK(new_inst.kind() == replaced[2].kind());
            CHECK(new_inst.qubits() == h_qubits_ref);
        }
    } else {
        CHECK(!atom->is_applicable(inst));
    }
}

// =============================================================================

class H2Rx
    : public decompositions::GateDecompositionRule<H2Rx, std::tuple<ops::H>, SINGLE_TGT_NO_CTRL, ops::parametric::Rx,
                                                   ops::parametric::Ph, ops::parametric::Ry> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0UL);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "H2Rx"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& /* op*/,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        assert(std::size(qubits) == 1);
        atom<ops::parametric::Rx>()->apply(circuit, ops::Rx{PI_VAL}, qubits);
        atom<ops::parametric::Ph>()->apply(circuit, ops::Ph{PI_VAL_2}, qubits);
        atom<ops::parametric::Ry>()->apply(circuit, ops::Ry{-PI_VAL_2}, qubits);
    }
};

// -----------------------------------------------------------------------------

TEST_CASE("GateDecompositionRule/H2Rx", "[decompositions][atom]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<::H2Rx>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    const auto h = ops::H();

    SECTION("No control qubits ") {
        qubits = {q0};
    }

    SECTION("Single control qubit") {
        qubits = {q0, q1};
    }

    SECTION("Double control qubit") {
        qubits = {q2, q0, q1};
    }

    SECTION("Triple control qubit") {
        qubits = {q3, q0, q2, q1};
    }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(h, qubits);
    const auto inst = original.instruction(inst_ref);

    REQUIRE(!inst.is_parametric());
    REQUIRE(atom->is_applicable(inst));
    atom->apply(decomposed, inst);

    CHECK(std::size(original) == 1);
    CHECK(std::size(decomposed) == 3);
    CHECK(check_unitary(original, decomposed));
}

// =============================================================================

class Rx2Rz
    : public decompositions::GateDecompositionRule<Rx2Rz, std::tuple<ops::Rx, ops::parametric::Rx>,
                                                   SINGLE_TGT_PARAM_NO_CTRL, ops::H, ops::parametric::Rz> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Rx2Rz"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& op,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        assert(std::size(qubits) == 1);
        atom<ops::H>()->apply(circuit, ops::H{}, qubits);
        std::visit(
            [this, &circuit, &qubits](const auto& param) {
                using param_t = std::remove_cvref_t<decltype(param)>;
                // Compile-time constant number of targets
                if constexpr (std::is_same_v<param_t, double>) {
                    atom<ops::parametric::Rz>()->apply(circuit, ops::Rz{param}, qubits);
                    return;
                } else if constexpr (std::is_same_v<param_t, param_list_t>) {
                    if (std::size(param) == 1) {
                        // TODO(dnguyen): This (.eval_smart()) should be taken care of by the apply() method...
                        atom<ops::parametric::Rz>()->apply(circuit, ops::parametric::Rz{param[0]}.eval_smart(), qubits);
                        return;
                    }
                }
                invalid_op_(circuit, qubits, param);
            },
            ops::parametric::get_param(op));

        atom<ops::H>()->apply(circuit, ops::H{}, qubits);
    }
};

// -----------------------------------------------------------------------------

TEST_CASE("GateDecompositionRule/Rx2Rz", "[decompositions][atom]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<::Rx2Rz>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("Numeric Rx") {
        auto rx = ops::Rx(52.2148);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            qubits = {q3, q0, q2, q1};
        }

        INFO("n_qubits = " << std::size(qubits));

        const auto inst_ref = original.apply_operator(rx, qubits);
        const auto inst = original.instruction(inst_ref);

        REQUIRE(inst.is_parametric());
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);

        CHECK(std::size(original) == 1);
        CHECK(std::size(decomposed) == 3);
        CHECK(check_unitary(original, decomposed));
    }

    SECTION("Parametric Rx") {
        auto rx = ops::parametric::Rx(52.2148);

        auto numeric = tweedledum::shallow_duplicate(original);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            qubits = {q3, q0, q2, q1};
        }
        INFO("n_qubits = " << std::size(qubits));

        numeric.apply_operator(rx.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(rx, qubits);
        const auto inst = original.instruction(inst_ref);

        REQUIRE(inst.is_parametric());
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);

        CHECK(std::size(original) == 1);
        CHECK(std::size(numeric) == 1);
        CHECK(std::size(decomposed) == 3);
        // NB: This works since the atom->apply() is calling eval_smart(...)
        CHECK(check_unitary(numeric, decomposed));
    }
}

// =============================================================================

class Ph2R
    : public decompositions::GateDecompositionRule<Ph2R, std::tuple<ops::Ph, ops::parametric::Ph>,
                                                   SINGLE_TGT_PARAM_SINGLE_CTRL, ops::parametric::P> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Ph2R"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::operator_t& op,
                    const mindquantum::qubits_t& qubits, const mindquantum::cbits_t& /* unused */) {
        std::visit(
            [this, &circuit, &qubits](const auto& param) {
                using param_t = std::remove_cvref_t<decltype(param)>;
                if constexpr (std::is_same_v<param_t, double>) {
                    atom<ops::parametric::P>()->apply(circuit, ops::P{param}, {qubits[0]});
                    return;
                } else if constexpr (std::is_same_v<param_t, param_list_t>) {
                    if (std::size(param) == 1) {
                        // TODO(dnguyen): This (.eval_smart()) should be taken care of by the apply() method...
                        atom<ops::parametric::P>()->apply(circuit, ops::parametric::P{param[0]}.eval_smart(),
                                                          {qubits[0]});
                        return;
                    }
                }
                invalid_op_(circuit, qubits, param);
            },
            ops::parametric::get_param(op));
    }
};

// -----------------------------------------------------------------------------

TEST_CASE("GateDecompositionRule/Ph2R", "[decompositions][atom]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<::Ph2R>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("Numeric Ph") {
        auto ph = ops::Ph(52.2148);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            qubits = {q3, q0, q2, q1};
        }

        const auto n_qubits = std::size(qubits);
        INFO("n_qubits = " << n_qubits);

        const auto inst_ref = original.apply_operator(ph, qubits);
        const auto inst = original.instruction(inst_ref);

        REQUIRE(inst.is_parametric());

        if (n_qubits == 1) {
            REQUIRE(!atom->is_applicable(inst));
        } else {
            REQUIRE(atom->is_applicable(inst));

            atom->apply(decomposed, inst);

            CHECK(std::size(original) == 1);
            CHECK(std::size(decomposed) == 1);
            CHECK(check_unitary(original, decomposed));
        }
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Ph(52.2148);

        auto numeric = tweedledum::shallow_duplicate(original);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            qubits = {q3, q0, q2, q1};
        }

        const auto n_qubits = std::size(qubits);
        INFO("n_qubits = " << n_qubits);

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        const auto inst = original.instruction(inst_ref);

        REQUIRE(inst.is_parametric());

        if (n_qubits == 1) {
            REQUIRE(!atom->is_applicable(inst));
        } else {
            REQUIRE(atom->is_applicable(inst));

            atom->apply(decomposed, inst);

            CHECK(std::size(original) == 1);
            CHECK(std::size(numeric) == 1);
            CHECK(std::size(decomposed) == 1);
            // NB: This works since the atom->apply() is calling eval_smart(...)
            CHECK(check_unitary(numeric, decomposed));
        }
    }
}

// =========================================================================
