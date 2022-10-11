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

#ifndef OPERATOR_TRAITS_HPP
#define OPERATOR_TRAITS_HPP

#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "experimental/core/config.hpp"
#include "experimental/core/traits.hpp"

namespace mindquantum::traits {
// -------------------------------------------------------------------------
// Helper traits to handle atoms that need to cover multiple kind of atoms

namespace impl {
template <typename ref_kind_t, typename... kinds_t>
constexpr bool kind_match(const ref_kind_t& ref_kind, kinds_t&&... kinds)
#if MQ_HAS_CONCEPTS
    requires(sizeof...(kinds_t) > 0)
#endif  // MQ_HAS_CONCEPTS
{
#if !MQ_HAS_CONCEPTS
    static_assert(sizeof...(kinds_t) > 0);
#endif  // MQ_HAS_CONCEPTS
    return ((ref_kind == std::forward<kinds_t>(kinds)) || ...);
}

template <typename ref_kind_t, typename>
struct tuple_kind_match;

template <typename ref_kind_t, typename... operators_t>
struct tuple_kind_match<ref_kind_t, std::tuple<operators_t...>> {
    static constexpr bool apply(const ref_kind_t& ref_kind) {
        return kind_match(ref_kind, operators_t::kind()...);
    }
};
}  // namespace impl

template <typename atom_t>
constexpr bool kind_compare(std::string_view kind) {
    using kinds_t = typename atom_t::kinds_t;
    static_assert(is_tuple_v<kinds_t>);
    return impl::tuple_kind_match<std::string_view, kinds_t>::apply(kind);
}

// =========================================================================

// If operator_t::num_targets exists, integral constant set to that number, else 0
template <typename op_t, typename = void>
struct static_variable_num_targets : std::integral_constant<std::size_t, 0UL> {};

/* NB: Really we should be able to get away with decltype(op_t::num_targets > 0)
 *     However, if -fms-extensions is given with GCC, then the above would not work for SFINAE anymore. This looks
 *     to be because some implicit conversion to integral is turned on by that flag if `op_t::num_targets`
 *     is a function pointer. The comparison with 0 seems to avoid that issue.
 */
template <typename op_t>
struct static_variable_num_targets<op_t, std::void_t<decltype(op_t::num_targets > 0)>>
    : std::integral_constant<std::size_t, op_t::num_targets> {};

// -------------------------------------------------------------------------

// If op_t::num_targets() exists, integral constant set to that number, else 0
template <typename op_t, typename = void>
struct static_method_num_targets : std::integral_constant<std::size_t, 0UL> {};

template <typename op_t>
#if MQ_HAS_CONCEPTS
struct static_method_num_targets<op_t, std::void_t<decltype(op_t::num_targets() > 0)>>
    : std::integral_constant<std::size_t, op_t::num_targets()> {
};
#else
struct static_method_num_targets<op_t, std::void_t<decltype(op_t::num_targets_static() > 0)>>
    : std::integral_constant<std::size_t, op_t::num_targets_static()> {
};
#endif  //  MQ_HAS_CONCEPTS

// -------------------------------------------------------------------------

// Map Eigen matrix types to number of qubits (0 if unknown)
template <typename matrix_t, typename = void>
struct matrix_const_num_rows : std::integral_constant<std::size_t, 0UL> {};

template <typename matrix_t>
struct matrix_const_num_rows<matrix_t, std::void_t<decltype(matrix_t::RowsAtCompileTime)>>
    : std::integral_constant<std::size_t, (matrix_t::RowsAtCompileTime >> 1UL)> {};

// If op_t has a matrix() method, deduce number of qubits from that, else 0
template <typename op_t, typename = void>
struct matrix_fixed_num_qubits : std::integral_constant<std::size_t, 0UL> {};

template <typename op_t>
struct matrix_fixed_num_qubits<op_t, std::void_t<decltype(std::declval<op_t>().matrix())>>
    : matrix_const_num_rows<std::remove_cvref_t<decltype(std::declval<op_t>().matrix())>>::type {};

// -------------------------------------------------------------------------

template <typename op_t, typename = void>
struct has_const_num_targets : std::true_type {};

template <typename op_t>
struct has_const_num_targets<op_t, std::void_t<typename op_t::non_const_num_targets>> : std::false_type {};

template <typename op_t>
inline constexpr auto has_const_num_targets_v = has_const_num_targets<op_t>::value;

// -------------------------------------------------------------------------

/* Deduce the number of qubits, either from:
 *   - op_t::num_targets (static constexpr attribute)
 *   - op_t::num_targets() (static potentially constexpr method)
 *   - return type of op_t::matrix() if constant number of rows
 *   - else 0
 */
// clang-format off
    template <typename op_t>
    inline constexpr auto num_targets = std::conditional_t<
        (static_variable_num_targets<op_t>::value > 0),
        typename static_variable_num_targets<op_t>::type,
        std::conditional_t<(static_method_num_targets<op_t>::value > 0),
                           typename static_method_num_targets<op_t>::type,
                           std::conditional_t<(matrix_fixed_num_qubits<op_t>::value > 0),
                                              typename matrix_fixed_num_qubits<op_t>::type,
                                              std::integral_constant<std::size_t, 0UL>>>
        >::value;
// clang-format on

// =========================================================================
}  // namespace mindquantum::traits

#endif /* OPERATOR_TRAITS_HPP */
