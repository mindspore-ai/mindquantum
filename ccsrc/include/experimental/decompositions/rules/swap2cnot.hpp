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

#ifndef DECOMPOSITION_RULE_SWAP2CNOT_HPP
#define DECOMPOSITION_RULE_SWAP2CNOT_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"

namespace mindquantum::decompositions::rules {
class Swap2CNOT : public GateDecompositionRule<Swap2CNOT, std::tuple<ops::Swap>, DUAL_TGT_ANY_CTRL, atoms::C<ops::X>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Swap2CNOT"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {qubits[0], qubits[1]});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {qubits[1], qubits[0]});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {qubits[0], qubits[1]});
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_SWAP2CNOT_HPP */
