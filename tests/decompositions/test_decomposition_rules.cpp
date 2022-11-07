//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#if __has_include("tweedledum/../../tests/check_unitary.h")
#    include "tweedledum/../../tests/check_unitary.h"
#else
#    include <tweedledum/test/check_unitary.h>
#endif
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/decompositions/atom_meta.hpp"
#include "experimental/decompositions/atom_storage.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/cnot2cz.hpp"
#include "experimental/decompositions/rules/cnot2rxx.hpp"
#include "experimental/decompositions/rules/crz2cxandrz.hpp"
#include "experimental/decompositions/rules/h2rx.hpp"
#include "experimental/decompositions/rules/no_control_ph.hpp"
#include "experimental/decompositions/rules/ph2r.hpp"
#include "experimental/decompositions/rules/qft2crandhadamard.hpp"
#include "experimental/decompositions/rules/r2rzandph.hpp"
#include "experimental/decompositions/rules/rx2rz.hpp"
#include "experimental/decompositions/rules/ry2rz.hpp"
#include "experimental/decompositions/rules/rz2rxandry.hpp"
#include "experimental/decompositions/rules/sqrtswap2cnotandsqrtx.hpp"
#include "experimental/decompositions/rules/swap2cnot.hpp"
#include "experimental/decompositions/rules/toffoli2cnotandtgate.hpp"
#include "experimental/decompositions/trivial_atom.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/gates/invalid.hpp"
#include "experimental/ops/gates/sqrtswap.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

namespace ops = mindquantum::ops;
namespace decompositions = mindquantum::decompositions;
namespace rules = mindquantum::decompositions::rules;

// ==============================================================================

namespace {
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

TEST_CASE("Decompositions/rules/CNOT2CZ", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::CNOT2CZ>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    // const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        qubits = {q0};
    }

    SECTION("Single control qubit") {
        do_smth = true;
        qubits = {q0, q1};
    }

    // NB: Tweedledum's check_unitary does fully not support control of controlled gates...

    // SECTION("Double control qubit") {
    //     do_smth = true;
    //     qubits = {q2, q0, q1};
    // }

    // SECTION("Triple control qubit") {
    //     do_smth = true;
    //     qubits = {q3, q0, q2, q1};
    // }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::X{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 3);
        CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/CNOT2Rxx", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::CNOT2Rxx>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    // const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        qubits = {q0};
    }

    SECTION("Single control qubit") {
        do_smth = true;
        qubits = {q0, q1};
    }

    // NB: Tweedledum's check_unitary does fully not support control of controlled gates...

    // SECTION("Double control qubit") {
    //     do_smth = true;
    //     qubits = {q2, q0, q1};
    // }

    // SECTION("Triple control qubit") {
    //     do_smth = true;
    //     qubits = {q3, q0, q2, q1};
    // }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::X{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 6);
        CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/CRz2CXAndRz", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    decompositions::DecompositionAtom atom(rules::CRZ2CXAndRz{storage});

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    bool do_smth(false);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto rz = ops::Rz(52.2148);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            do_smth = true;
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            do_smth = true;
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            do_smth = true;
            qubits = {q3, q0, q2, q1};
        }

        numeric.apply_operator(rz, qubits);
        const auto inst_ref = original.apply_operator(rz, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto rz = ops::parametric::Rz(52.2148);

        SECTION("No control qubits ") {
            qubits = {q0};
        }

        SECTION("Single control qubit") {
            do_smth = true;
            qubits = {q0, q1};
        }

        SECTION("Double control qubit") {
            do_smth = true;
            qubits = {q2, q0, q1};
        }

        SECTION("Triple control qubit") {
            do_smth = true;
            qubits = {q3, q0, q2, q1};
        }

        numeric.apply_operator(rz.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(rz, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());

    if (!do_smth) {
        REQUIRE(!atom.is_applicable(inst));
    } else {
        REQUIRE(atom.is_applicable(inst));

        atom.apply(decomposed, inst);

        CHECK(std::size(original) == 1);
        CHECK(std::size(numeric) == 1);
        CHECK(std::size(decomposed) == 4);
        // NB: This works since atom->apply() is calling eval_smart(...)
        CHECK(check_unitary(numeric, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/H2Rx", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::H2Rx>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        qubits = {q0};
    }

    SECTION("Single control qubit") {
        do_smth = true;
        qubits = {q0, q1};
    }

    SECTION("Double control qubit") {
        do_smth = true;
        qubits = {q2, q0, q1};
    }

    SECTION("Triple control qubit") {
        do_smth = true;
        qubits = {q3, q0, q2, q1};
    }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::H{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 3);
        CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/ControlPh", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    decompositions::DecompositionAtom atom(rules::RemovePhNoCtrl{storage});

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto ph = ops::Ph(52.2148);

        SECTION("No control qubits ") {
            do_smth = true;
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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Ph(52.2148);

        SECTION("No control qubits ") {
            do_smth = true;
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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());

    CHECK(std::size(original) == 1);
    if (do_smth) {
        REQUIRE(atom.is_applicable(inst));
        atom.apply(decomposed, inst);
        CHECK(std::size(decomposed) == 0);
    } else {
        REQUIRE(!atom.is_applicable(inst));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/Ph2R", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Ph2R>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Ph(52.2148);

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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());

    if (n_qubits == 1) {
        REQUIRE(!atom->is_applicable(inst));
    } else {
        REQUIRE(atom->is_applicable(inst));

        atom->apply(decomposed, inst);

        CHECK(std::size(original) == 1);
        CHECK(std::size(numeric) == 1);
        CHECK(std::size(decomposed) == 1);
        // NB: This works since atom->apply() is calling eval_smart(...)
        CHECK(check_unitary(numeric, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/R2RzAndPh", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::R2RzAndPh>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto ph = ops::P(52.2148);

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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::P(52.2148);

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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());
    REQUIRE(atom->is_applicable(inst));

    atom->apply(decomposed, inst);

    CHECK(std::size(original) == 1);
    CHECK(std::size(numeric) == 1);
    CHECK(std::size(decomposed) == 2);
    // NB: This works since atom->apply() is calling eval_smart(...)
    CHECK(check_unitary(numeric, decomposed));
}

// =============================================================================

TEST_CASE("Decompositions/rules/Rx2Rz", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Rx2Rz>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto ph = ops::Rx(52.2148);

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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Rx(52.2148);

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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());
    REQUIRE(atom->is_applicable(inst));

    atom->apply(decomposed, inst);

    CHECK(std::size(original) == 1);
    CHECK(std::size(numeric) == 1);
    CHECK(std::size(decomposed) == 3);
    // NB: This works since atom->apply() is calling eval_smart(...)
    CHECK(check_unitary(numeric, decomposed));
}

// =============================================================================

TEST_CASE("Decompositions/rules/Ry2Rz", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Ry2Rz>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto ph = ops::Ry(52.2148);

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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Ry(52.2148);

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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());
    REQUIRE(atom->is_applicable(inst));

    atom->apply(decomposed, inst);

    CHECK(std::size(original) == 1);
    CHECK(std::size(numeric) == 1);
    CHECK(std::size(decomposed) == 3);
    // NB: This works since atom->apply() is calling eval_smart(...)
    CHECK(check_unitary(numeric, decomposed));
}

// =============================================================================

TEST_CASE("Decompositions/rules/Rz2RxAndRy", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Rz2RxAndRy>();

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);
    auto numeric = tweedledum::shallow_duplicate(original);

    instruction_t inst(ops::Invalid{}, {q0}, {});

    SECTION("Numeric Ph") {
        auto ph = ops::Rz(52.2148);

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

        numeric.apply_operator(ph, qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    SECTION("Parametric Ph") {
        auto ph = ops::parametric::Rz(52.2148);

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

        numeric.apply_operator(ph.eval_full(), qubits);
        const auto inst_ref = original.apply_operator(ph, qubits);
        inst = original.instruction(inst_ref);
    }

    const auto n_qubits = std::size(qubits);
    INFO("n_qubits = " << n_qubits);

    REQUIRE(inst.is_parametric());
    REQUIRE(atom->is_applicable(inst));

    atom->apply(decomposed, inst);

    CHECK(std::size(original) == 1);
    CHECK(std::size(numeric) == 1);
    CHECK(std::size(decomposed) == 3);
    // NB: This works since atom->apply() is calling eval_smart(...)
    CHECK(check_unitary(numeric, decomposed));
}

// =============================================================================

TEST_CASE("Decompositions/rules/SqrtSwap2CNOT", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::SqrtSwap2CNOTAndSqrtX>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        do_smth = true;
        qubits = {q0, q1};
    }

    // NB: Tweedledum's check_unitary does fully not support controlled multi-qubit gates...

    // SECTION("Single control qubit") {
    //     do_smth = true;
    //     qubits = {q2, q0, q1};
    // }

    // SECTION("Double control qubit") {
    //     do_smth = true;
    //     qubits = {q3, q0, q2, q1};
    // }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::SqrtSwap{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 3);
        CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/Swap2CNOT", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Swap2CNOT>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        do_smth = true;
        qubits = {q0, q1};
    }

    // NB: Tweedledum's check_unitary does fully not support controlled multi-qubit gates...

    // SECTION("Single control qubit") {
    //     do_smth = true;
    //     qubits = {q2, q0, q1};
    // }

    // SECTION("Double control qubit") {
    //     do_smth = true;
    //     qubits = {q3, q0, q2, q1};
    // }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::Swap{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 3);
        CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================

TEST_CASE("Decompositions/rules/Toffoli2CNOTAndT", "[decompositions][rules]") {
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using circuit_t = tweedledum::Circuit;

    decompositions::AtomStorage storage;
    auto* atom = storage.add_or_replace_atom<rules::Toffoli2CNOTAndT>();

    auto do_smth = false;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    const auto q3 = original.create_qubit();

    mindquantum::qubits_t qubits;
    auto decomposed = tweedledum::shallow_duplicate(original);

    SECTION("No control qubits ") {
        qubits = {q0};
    }

    SECTION("Single control qubit") {
        qubits = {q0, q1};
    }

    SECTION("Double control qubit") {
        do_smth = true;
        qubits = {q0, q1, q2};
    }

    // NB: Tweedledum's check_unitary does not fully support controlled gates...

    // SECTION("Triple control qubit") {
    //     do_smth = true;
    //     qubits = {q3, q0, q2, q1};
    // }

    INFO("n_qubits = " << std::size(qubits));

    const auto inst_ref = original.apply_operator(ops::X{}, qubits);
    const auto inst = original.instruction(inst_ref);

    CHECK(std::size(original) == 1);
    REQUIRE(!inst.is_parametric());
    if (do_smth) {
        REQUIRE(atom->is_applicable(inst));
        atom->apply(decomposed, inst);
        CHECK(std::size(decomposed) == 15);
        // TODO(dnguyen): Fix failing test!
        // CHECK(check_unitary(original, decomposed));
    }
}

// =============================================================================
