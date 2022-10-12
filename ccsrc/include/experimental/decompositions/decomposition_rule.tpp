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

#ifndef DECOMPOSITION_RULE_TPP
#define DECOMPOSITION_RULE_TPP

#include <tuple>

#include "experimental/decompositions/config.hpp"
#include "experimental/decompositions/decomposition_atom.hpp"
#ifndef DECOMPOSITION_RULE_HPP
#    error This file must only be included by decomposition_rule.hpp!
#endif  // DECOMPOSITION_RULE_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by atom_storage.hpp
#include "experimental/decompositions/decomposition_rule.hpp"
#include "experimental/ops/gates/invalid.hpp"

namespace mindquantum::decompositions {

// =========================================================================
// ::DecompositionRule()

template <typename derived_t, typename... atoms_t>
DecompositionRule<derived_t, atoms_t...>::DecompositionRule(AtomStorage& storage)
    : atoms_{create_<atoms_t...>(storage)} {
}

// =========================================================================
// ::atom()

template <typename derived_t, typename... atoms_t>
template <std::size_t idx>
constexpr auto* DecompositionRule<derived_t, atoms_t...>::atom() noexcept MQ_REQUIRES((idx < sizeof...(atoms_t))) {
#if !MQ_HAS_CONCEPTS
    static_assert(idx < sizeof...(atoms_t));
#endif  // !MQ_HAS_CONCEPTS
    return atoms_[idx];
}

template <typename derived_t, typename... atoms_t>
template <typename atom_t>
constexpr auto* DecompositionRule<derived_t, atoms_t...>::atom() noexcept
    MQ_REQUIRES((concepts::tuple_contains<typename traits::atom_traits<atom_t>::type,
                                          typename traits::atom_traits<atoms_t>::type...>) ) {
    using real_atom_t = typename traits::atom_traits<atom_t>::type;
    using atoms_tuple_t = std::tuple<typename traits::atom_traits<atoms_t>::type...>;
#if !MQ_HAS_CONCEPTS
    static_assert(traits::tuple_contains<real_atom_t, atoms_tuple_t>);
#endif  // !MQ_HAS_CONCEPTS
    return atom<details::index_in_tuple<real_atom_t, atoms_tuple_t>>();
}

// =========================================================================
// ::apply()

template <typename derived_t, typename... atoms_t>
void DecompositionRule<derived_t, atoms_t...>::apply(circuit_t& circuit, const operator_t& op, const qubits_t& qubits,
                                                     const cbits_t& cbits) noexcept {
    static_cast<derived_t*>(this)->apply_impl(circuit, op, qubits, cbits);
}

// =========================================================================
// ::invalid_op_()

template <typename derived_t, typename... atoms_t>
void DecompositionRule<derived_t, atoms_t...>::invalid_op_(circuit_t& circuit, const qubits_t& qubits) {
    circuit.apply_operator(ops::Invalid{std::size(qubits)}, qubits);
}

template <typename derived_t, typename... atoms_t>
void DecompositionRule<derived_t, atoms_t...>::invalid_op_(circuit_t& circuit, const qubits_t& qubits,
                                                           const gate_param_t& /* param */) {
    invalid_op_(circuit, qubits);
}

// =========================================================================
// ::create_()

template <typename derived_t, typename... atoms_t>
template <typename... args_t>
auto DecompositionRule<derived_t, atoms_t...>::create_(AtomStorage& storage) {
    return std::array<DecompositionAtom*, sizeof...(args_t)>{create_el_<args_t>(storage)...};
}

// =========================================================================
// ::create_el()

template <typename derived_t, typename... atoms_t>
template <typename T>
auto DecompositionRule<derived_t, atoms_t...>::create_el_(AtomStorage& storage) {
    using atom_t = typename traits::atom_traits<T>::type;
    return storage.add_or_compatible_atom<atom_t>();
}

}  // namespace mindquantum::decompositions

#endif /* DECOMPOSITION_RULE_TPP */
