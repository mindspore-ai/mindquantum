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

#ifndef DECOMPOSITION_RULE_PH2R_HPP
#define DECOMPOSITION_RULE_PH2R_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

namespace mindquantum::decompositions::rules {
class Ph2R
    : public GateDecompositionRule<Ph2R, std::tuple<ops::Ph, ops::parametric::Ph>, SINGLE_TGT_PARAM_SINGLE_CTRL,
                                   ops::parametric::P> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Ph2R"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& op, const qubits_t& qubits, const cbits_t& /* unused */) {
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
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_PH2R_HPP */
