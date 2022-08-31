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

#ifndef GATE_DECOMPOSITION_RULE_CXX20_TPP
#define GATE_DECOMPOSITION_RULE_CXX20_TPP

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#include "experimental/decompositions/details/traits.hpp"
#ifndef GATE_DECOMPOSITION_RULE_CXX20_HPP
#    error This file must only be included by gate_decomposition_rule_cxx20.hpp!
#endif  // GATE_DECOMPOSITION_RULE_CXX20_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by atom_storage.hpp
#include "experimental/decompositions/details/gate_decomposition_rule_cxx20.hpp"

namespace mindquantum::decompositions {

// =========================================================================
// ::is_compatible()

template <typename derived_t, typename kinds_t, DecompositionRuleParam param_, typename... atoms_t>
template <typename rule_t>
MQ_NODISCARD constexpr bool GateDecompositionRule<derived_t, kinds_t, param_, atoms_t...>::is_compatible()
    const noexcept {
    return param_.num_targets == rule_t::num_targets_
           && (param_.num_controls < 0 || rule_t::num_controls_ < 0 || param_.num_controls == rule_t::num_controls_);
}

// =========================================================================
// ::is_applicable()

template <typename derived_t, typename kinds_t, DecompositionRuleParam param_, typename... atoms_t>
MQ_NODISCARD bool GateDecompositionRule<derived_t, kinds_t, param_, atoms_t...>::is_applicable(
    const instruction_t& inst) const noexcept {
    if constexpr (traits::has_is_applicable_v<derived_t>) {
        return static_cast<const derived_t*>(this)->is_applicable_impl(inst);
    } else {
        return traits::kind_compare<derived_t>(inst.kind()) && (derived_t::num_params() == inst.num_parameters())
               && ((derived_t::num_targets() == decompositions::any_target)
                   || derived_t::num_targets() == inst.num_targets())
               && (derived_t::num_controls() == decompositions::any_control
                   /* It is ok for a decomposition rule constrained on N control qubits to decompose an instruction
                    * with M >= N qubits; the "extra" control qubits are simply counted as "free" control qubits.
                    */
                   || derived_t::num_controls() <= inst.num_controls());  // TODO(dnguyen): Use == instead?
    }
}
}  // namespace mindquantum::decompositions

#endif /* GATE_DECOMPOSITION_RULE_CXX20_TPP */
