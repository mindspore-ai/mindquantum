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

#ifndef DECOMPOSITION_RULE_CNU2TOFFOLIANDCU_HPP
#define DECOMPOSITION_RULE_CNU2TOFFOLIANDCU_HPP

#include <algorithm>
#include <iterator>

#include "experimental/core/compute.hpp"
#include "experimental/core/control.hpp"
#include "experimental/decompositions/non_gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

namespace mindquantum::decompositions::rules {
class CNu2ToffoliAndCu : public decompositions::NonGateDecompositionRule<CNu2ToffoliAndCu, atoms::C<ops::X, 2>> {
 public:
    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "CNu2ToffoliAndCu"sv;
    }

    MQ_NODISCARD static bool is_applicable(const decompositions::instruction_t& inst) {
        return inst.num_controls() > 2 || (inst.num_controls() == 2 && inst.kind() != ops::X::kind());
    }

    void apply_impl(circuit_t& circuit, const instruction_t& inst, const qubits_t& qubits) {
        const auto& kind = inst.kind();
        auto n_controls = inst.num_controls();

        qubits_t controls;
        std::copy(begin(qubits), begin(qubits) + n_controls, std::back_inserter(controls));

        qubits_t targets;
        std::copy(begin(qubits) + n_controls, end(qubits), std::back_inserter(targets));

        const auto is_x_larger_than_toffoli = kind == ops::X::kind() && n_controls > 2;
        if (is_x_larger_than_toffoli) {
            --n_controls;
        }

        qubits_t ancillae;
        for (auto i(0UL); i < n_controls - 1; ++i) {
            ancillae.emplace_back(circuit.create_qubit());
        }

        MQ_WITH_COMPUTE(circuit, compute) {
            atom<atoms::C<ops::X, 2>>()->apply(compute, ops::X{}, {controls[0], controls[1], ancillae[0]});
            for (auto ctrl_idx(2UL); ctrl_idx < n_controls; ++ctrl_idx) {
                atom<atoms::C<ops::X, 2>>()->apply(
                    compute, ops::X{}, {controls[ctrl_idx], ancillae[ctrl_idx - 2], ancillae[ctrl_idx - 1]});
            }
        }
        MQ_WITH_COMPUTE_END

        qubits_t ctrls{ancillae.back()};

        if (is_x_larger_than_toffoli) {
            ctrls.emplace_back(controls.back());
        }

        MQ_WITH_CONTROL(circuit, controlled, ctrls) {
            const auto new_inst = instruction_t{inst, targets, {}};
            if (auto* atom{storage().get_atom_for(new_inst)}; atom) {
                atom->apply(circuit, new_inst);
            } else {
                controlled.apply_operator(new_inst);
            }
        }

        // Automatic uncompute
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_CNU2TOFFOLIANDCU_HPP */
