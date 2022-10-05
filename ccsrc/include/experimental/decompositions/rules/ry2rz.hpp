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

#ifndef DECOMPOSITION_RULE_RY2RZ_HPP
#define DECOMPOSITION_RULE_RY2RZ_HPP
#include <tuple>

#include <symengine/constants.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/parametric/register_gate_type.hpp"

namespace mindquantum::decompositions::rules {
class Ry2Rz
    : public GateDecompositionRule<Ry2Rz, std::tuple<ops::Ry, ops::parametric::Ry>, SINGLE_TGT_PARAM_ANY_CTRL,
                                   ops::parametric::Rx, ops::parametric::Rz> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Ry2Rz"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& op, const qubits_t& qubits, const cbits_t& /* unused */) {
        atom<ops::parametric::Rx>()->apply(circuit, ops::Rx{PI_VAL_2}, qubits);
        std::visit(
            [this, &circuit, &qubits](const auto& param) {
                using ops::parametric::param_list_t;
                using param_t = std::remove_cvref_t<decltype(param)>;

                if constexpr (std::is_same_v<param_t, double>) {
                    atom<ops::parametric::Rz>()->apply(circuit, ops::Rz{param}, qubits);
                    return;
                } else if constexpr (std::is_same_v<param_t, param_list_t>) {
                    if (std::size(param) == 1) {
                        atom<ops::parametric::Rz>()->apply(circuit, ops::parametric::Rz{param[0]}.eval_smart(), qubits);
                        return;
                    }
                }
                invalid_op_(circuit, qubits, param);
            },
            ops::parametric::get_param(op));
        atom<ops::parametric::Rx>()->apply(circuit, ops::Rx{-PI_VAL_2}, qubits);
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_RY2RZ_HPP */
