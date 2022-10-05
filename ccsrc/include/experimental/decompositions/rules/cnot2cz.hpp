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

#ifndef DECOMPOSITION_RULE_CNOT2CZ_HPP
#define DECOMPOSITION_RULE_CNOT2CZ_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"

namespace mindquantum::decompositions::rules {
class CNOT2CZ
    : public GateDecompositionRule<CNOT2CZ, std::tuple<ops::X>, SINGLE_TGT_SINGLE_CTRL, ops::H, atoms::C<ops::Z>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "CNOT2CZ"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[1]});
        atom<atoms::C<ops::Z>>()->apply(circuit, ops::Z{}, {qubits[0], qubits[1]});
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[1]});
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_CNOT2CZ_HPP */
