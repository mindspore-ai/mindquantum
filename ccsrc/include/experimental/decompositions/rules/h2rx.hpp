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

#ifndef DECOMPOSITION_RULE_H2RX_HPP
#define DECOMPOSITION_RULE_H2RX_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

namespace mindquantum::decompositions::rules {
class H2Rx
    : public GateDecompositionRule<H2Rx, std::tuple<ops::H>, SINGLE_TGT_NO_CTRL, ops::parametric::Rx,
                                   ops::parametric::Ph, ops::parametric::Ry> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0UL);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "H2Rx"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        assert(std::size(qubits) == 1);
        atom<ops::parametric::Rx>()->apply(circuit, ops::Rx{PI_VAL}, qubits);
        atom<ops::parametric::Ph>()->apply(circuit, ops::Ph{PI_VAL_2}, qubits);
        atom<ops::parametric::Ry>()->apply(circuit, ops::Ry{-PI_VAL_2}, qubits);
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_H2RX_HPP */
