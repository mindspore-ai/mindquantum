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

#ifndef DECOMPOSITION_RULE_ENTANGLE_HPP
#define DECOMPOSITION_RULE_ENTANGLE_HPP

#include <algorithm>
#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"

namespace mindquantum::decompositions::rules {
class Entangle2HAndCNOT
    : public GateDecompositionRule<Entangle2HAndCNOT, std::tuple<ops::Entangle>, ANY_TGT_NO_CTRL, ops::H,
                                   atoms::C<ops::X>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Entangle2HAndCNOT"sv;
    }

    void apply_impl(circuit_t& circuit, const decompositions::operator_t& /* op */,
                    const decompositions::qubits_t& qubits, const decompositions::cbits_t& /* unused */) {
        atom<ops::H>()->apply(circuit, ops::H{}, {qubits[0]});

        auto tgt{qubits.front()};

        std::for_each(begin(qubits) + 1, end(qubits), [&circuit, &tgt, this](const qubit_t& qubit) {
            atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {tgt, qubit});
        });
    }
};
}  // namespace mindquantum::decompositions::rules
#endif /* DECOMPOSITION_RULE_ENTANGLE_HPP */
