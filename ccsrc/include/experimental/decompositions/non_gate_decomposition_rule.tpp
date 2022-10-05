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

#ifndef NON_GATE_DECOMPOSITION_RULE_TPP
#define NON_GATE_DECOMPOSITION_RULE_TPP

#ifndef NON_GATE_DECOMPOSITION_RULE_HPP
#    error This file must only be included by non_gate_decomposition_rule.hpp!
#endif  // NON_GATE_DECOMPOSITION_RULE_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by non_gate_decomposition_rule.hpp
#include <tuple>
#include <utility>

#include "experimental/decompositions/non_gate_decomposition_rule.hpp"

namespace mindquantum::decompositions {

// =========================================================================
// ::NonGateDecompositionRule

template <typename derived_t, typename... atoms_t>
template <typename atom_t, typename... args_t>
constexpr auto* NonGateDecompositionRule<derived_t, atoms_t...>::atom(args_t&&... args) noexcept {
    using real_atom_t = typename traits::atom_traits<atom_t>::type;
    using atoms_tuple_t = std::tuple<typename traits::atom_traits<atoms_t>::type...>;

    if constexpr (traits::tuple_contains<real_atom_t, atoms_tuple_t>) {
        return parent_t::template atom<real_atom_t>();
    } else {
        return storage_.add_or_return_atom<real_atom_t>(std::forward<args_t>(args)...);
    }
}

// =========================================================================
// ::apply()

template <typename derived_t, typename... atoms_t>
void NonGateDecompositionRule<derived_t, atoms_t...>::apply(circuit_t& circuit, const instruction_t& inst) noexcept {
    static_cast<derived_t*>(this)->apply_impl(circuit, inst);
}

}  // namespace mindquantum::decompositions

#endif /* NON_GATE_DECOMPOSITION_RULE_TPP */
