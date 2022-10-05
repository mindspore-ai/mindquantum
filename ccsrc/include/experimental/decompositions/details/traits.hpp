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

#ifndef DECOMPOSITION_RULES_TRAITS_HPP
#define DECOMPOSITION_RULES_TRAITS_HPP

#include <type_traits>
#include <utility>

#include "experimental/core/config.hpp"
#include "experimental/decompositions/config.hpp"
#include "experimental/ops/parametric/traits.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/gate_concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

#include "experimental/decompositions/details/decomposition_param.hpp"
#include "experimental/decompositions/parametric_atom.hpp"
#include "experimental/decompositions/trivial_atom.hpp"

namespace mindquantum::decompositions {
#if MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
template <typename derived_t, typename kinds_t, DecompositionRuleParam param, typename... atoms_t>
class GateDecompositionRule;
#else
template <typename derived_t, typename kinds_t, uint32_t, num_control_t, uint32_t, typename... atoms_t>
class GateDecompositionRuleCXX17;
#endif  // MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
}  // namespace mindquantum::decompositions

namespace mindquantum::traits {
// Automatic type conversion utilities
#if MQ_HAS_CONCEPTS
template <typename atom_t>
struct atom_traits;

template <concepts::ParametricGate gate_t>
struct atom_traits<gate_t> {
    using type = decompositions::ParametricSimpleAtom<gate_t>;
};
template <concepts::NonParametricGate gate_t>
struct atom_traits<gate_t> {
    using type = decompositions::TrivialSimpleAtom<gate_t>;
};
#else
namespace details {
template <typename T, typename = void>
struct helper : std::false_type {};

template <typename T>
struct helper<T, std::void_t<decltype(std::declval<T>().params())>> : std::true_type {};
}  // namespace details

template <typename atom_t>
struct atom_traits {
    using type = std::conditional_t<details::helper<atom_t>::value, decompositions::ParametricSimpleAtom<atom_t>,
                                    decompositions::TrivialSimpleAtom<atom_t>>;
};
#endif

#if MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS
template <typename derived_t, typename kinds_t, decompositions::DecompositionRuleParam param, typename... atoms_t>
struct atom_traits<decompositions::GateDecompositionRule<derived_t, kinds_t, param, atoms_t...>> {
    using type = decompositions::GateDecompositionRule<derived_t, kinds_t, param, atoms_t...>;
};
#else
template <typename derived_t, typename kinds_t, uint32_t num_targets, decompositions::num_control_t num_controls,
          uint32_t num_params, typename... atoms_t>
struct atom_traits<decompositions::GateDecompositionRuleCXX17<derived_t, kinds_t, num_targets, num_controls, num_params,
                                                              kinds_t, atoms_t...>> {
    using type = decompositions::GateDecompositionRuleCXX17<derived_t, kinds_t, num_targets, num_controls, num_params,
                                                            kinds_t, atoms_t...>;
};
#endif  // MQ_HAS_CLASS_NON_TYPE_TEMPLATE_ARGS

template <typename op_t, uint32_t num_targets, decompositions::num_control_t num_controls>
struct atom_traits<decompositions::TrivialAtom<op_t, num_targets, num_controls>> {
    using type = decompositions::TrivialAtom<op_t, num_targets, num_controls>;
};

template <typename op_t, uint32_t num_targets, decompositions::num_control_t num_controls>
struct atom_traits<decompositions::ParametricAtom<op_t, num_targets, num_controls>> {
    using type = decompositions::ParametricAtom<op_t, num_targets, num_controls>;
};

// -------------------------------------------------------------------------
// Helper traits to handle atoms that do not provide atom_t::is_applicable

template <typename atom_t, typename = void>
struct has_is_applicable : std::false_type {};

template <typename atom_t>
struct has_is_applicable<
    atom_t, std::void_t<decltype(std::declval<atom_t>().is_applicable_impl(std::declval<instruction_t>()))>>
    : std::true_type {};

template <typename atom_t>
inline constexpr auto has_is_applicable_v = has_is_applicable<atom_t>::value;
}  // namespace mindquantum::traits

namespace mindquantum::decompositions::details {
template <std::size_t idx, typename elem_t, typename tuple_t>
constexpr auto index_in_tuple_fn()
#if MQ_HAS_CONCEPTS
    requires(idx < std::tuple_size_v<tuple_t> /* if this fails, elem_t is not in tuple_t */)
#endif  // MQ_HAS_CONCEPTS
{
#if !MQ_HAS_CONCEPTS
    static_assert(idx < std::tuple_size_v<tuple_t>, "The element is not in the tuple!");
#endif  // !MQ_HAS_CONCEPTS
    using tuple_elem_t = typename std::tuple_element_t<idx, tuple_t>;
    if constexpr (std::is_same_v<elem_t, tuple_elem_t>) {
        return idx;
    } else {
        return index_in_tuple_fn<idx + 1, elem_t, tuple_t>();
    }
}

template <typename elem_t, typename tuple_t>
inline constexpr auto index_in_tuple = index_in_tuple_fn<0, elem_t, tuple_t>();
}  // namespace mindquantum::decompositions::details

#endif /* DECOMPOSITION_RULES_TRAITS_HPP */
