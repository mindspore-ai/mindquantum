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

#ifndef DECOMPOSITION_RULE_CRZ2CXANDRZ_HPP
#define DECOMPOSITION_RULE_CRZ2CXANDRZ_HPP

#include <algorithm>

#include <symengine/mul.h>
#include <symengine/real_double.h>

#include "experimental/decompositions/non_gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"
#include "experimental/ops/parametric/config.hpp"
#include "experimental/ops/parametric/register_gate_type.hpp"

namespace mindquantum::decompositions::rules {
class CRZ2CXAndRz : public decompositions::NonGateDecompositionRule<CRZ2CXAndRz, ops::parametric::Rz, ops::X> {
 public:
    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "CRZ2CXAndRz"sv;
    }

    MQ_NODISCARD static bool is_applicable(const instruction_t& inst) {
        return inst.is_one<ops::Rz, ops::parametric::Rz>() && inst.num_controls() > 0;
    }

    void apply_impl(circuit_t& circuit, const instruction_t& inst) {
        const auto& kind = inst.kind();
        const auto& qubits = inst.qubits();
        auto n_controls = inst.num_controls();

        qubits_t cnot_qubits;
        std::copy(begin(qubits), begin(qubits) + n_controls, std::back_inserter(cnot_qubits));
        cnot_qubits.emplace_back(inst.target());

        qubits_t targets{inst.target()};

        std::visit(
            [this, &circuit, &targets, &cnot_qubits](const auto& param) {
                using ops::parametric::param_list_t;
                using param_t = std::remove_cvref_t<decltype(param)>;
                if constexpr (std::is_same_v<param_t, double>) {
                    atom<ops::parametric::Rz>()->apply(circuit, ops::Rz{0.5 * param}, targets);
                } else if constexpr (std::is_same_v<param_t, param_list_t>) {
                    if (std::size(param) == 1) {
                        atom<ops::parametric::Rz>()->apply(
                            circuit, ops::parametric::Rz{SymEngine::mul(param[0], SymEngine::number(0.5))}.eval_smart(),
                            targets);
                    }
                } else {
                    invalid_op_(circuit, targets, param);
                }

                atom<ops::X>()->apply(circuit, ops::X{}, cnot_qubits);

                if constexpr (std::is_same_v<param_t, double>) {
                    atom<ops::parametric::Rz>()->apply(circuit, ops::Rz{-0.5 * param}, targets);
                } else if constexpr (std::is_same_v<param_t, param_list_t>) {
                    if (std::size(param) == 1) {
                        atom<ops::parametric::Rz>()->apply(
                            circuit,
                            ops::parametric::Rz{SymEngine::mul(param[0], SymEngine::number(-0.5))}.eval_smart(),
                            targets);
                    }
                } else {
                    invalid_op_(circuit, targets, param);
                }

                atom<ops::X>()->apply(circuit, ops::X{}, cnot_qubits);
            },
            ops::parametric::get_param(inst));
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_CRZ2CXANDRZ_HPP */
