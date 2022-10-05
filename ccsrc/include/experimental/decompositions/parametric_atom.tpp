//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef PARAMETRIC_ATOM_TPP
#define PARAMETRIC_ATOM_TPP

#include "experimental/decompositions/config.hpp"
#include "experimental/ops/parametric/substitutions.hpp"

#ifndef PARAMETRIC_ATOM_HPP
#    error This file must only be included by parametric_atom.hpp!
#endif  // PARAMETRIC_ATOM_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by parametric_atom.hpp
#include "experimental/decompositions/parametric_atom.hpp"

namespace mindquantum::decompositions {
template <typename op_t, num_target_t num_targets_, num_control_t num_controls_>
bool ParametricAtom<op_t, num_targets_, num_controls_>::is_applicable(const instruction_t& inst) const noexcept {
    return ((op_t::kind() == inst.kind() || op_t::non_param_type::kind() == inst.kind())  // Technically not required?
            && op_t::num_params == inst.num_parameters()
            && ((num_targets_ == any_target) || num_targets_ == inst.num_targets())
            && (num_controls_ == any_control || num_controls_ == inst.num_controls()));
}

// =========================================================================

template <typename op_t, num_target_t num_targets_, num_control_t num_controls_>
void ParametricAtom<op_t, num_targets_, num_controls_>::apply(circuit_t& circuit, const operator_t& op,
                                                              const qubits_t& qubits, const cbits_t& cbits) noexcept {
    circuit.apply_operator(op, qubits, cbits);
}
}  // namespace mindquantum::decompositions

#endif /* PARAMETRIC_ATOM_TPP */
