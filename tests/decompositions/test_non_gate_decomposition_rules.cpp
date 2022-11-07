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

#include <string>
#include <string_view>

#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "mindquantum/catch2/mindquantum.hpp"
#include "mindquantum/catch2/tweedledum.hpp"

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/non_gate_decomposition_rule.hpp"
#include "experimental/ops/gates.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

using mindquantum::catch2::Equals;
namespace ops = mindquantum::ops;
namespace decompositions = mindquantum::decompositions;

// =============================================================================

namespace {
class AllToXAndY : public decompositions::NonGateDecompositionRule<AllToXAndY, ops::X> {
 public:
    using base_t::base_t;

    static constexpr auto kind() noexcept {
        return ops::X::kind();
    }

    static constexpr auto name() noexcept {
        return "CNOT2CZ"sv;
    }

    void apply_impl(mindquantum::circuit_t& circuit, const mindquantum::instruction_t& inst) {
        atom<ops::X>()->apply(circuit, ops::X{}, inst.qubits());
        atom<ops::Y>()->apply(circuit, ops::Y{}, inst.qubits());
    }
};
}  // namespace

// -----------------------------------------------------------------------------

TEST_CASE("NonGateDecompositionRule/atom", "[decompositions][atom]") {
    using instruction_t = mindquantum::instruction_t;
    using circuit_t = mindquantum::circuit_t;

    using x_t = decompositions::TrivialSimpleAtom<ops::X>;
    using y_t = decompositions::TrivialSimpleAtom<ops::Y>;

    decompositions::AtomStorage storage;
    circuit_t circuit;

    circuit_t original;
    const auto q0 = original.create_qubit();
    const auto q1 = original.create_qubit();
    const auto q2 = original.create_qubit();
    auto ref0 = original.apply_operator(ops::H{}, {q0});
    auto ref1 = original.apply_operator(ops::Z{}, {q0, q1});

    const auto inst0 = original.instruction(ref0);
    const auto inst1 = original.instruction(ref1);

    circuit_t reference(tweedledum::shallow_duplicate(original));
    reference.apply_operator(ops::X(), {q0});
    reference.apply_operator(ops::Y(), {q0});
    reference.apply_operator(ops::X(), {q0, q1});
    reference.apply_operator(ops::Y(), {q0, q1});

    ::AllToXAndY rule(storage);

    CHECK(storage.has_atom<x_t>());
    CHECK(!storage.has_atom<y_t>());

    circuit_t decomposed(tweedledum::shallow_duplicate(original));
    original.foreach_instruction([&decomposed, &rule](const instruction_t& inst) { rule.apply(decomposed, inst); });

    CHECK(storage.has_atom<x_t>());
    CHECK(storage.has_atom<y_t>());
    CHECK_THAT(decomposed, Equals(reference));
}
