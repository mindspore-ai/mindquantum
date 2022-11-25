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

#ifndef PARAMETER_RESOLVER_EXTERNAL_OPS_HPP
#define PARAMETER_RESOLVER_EXTERNAL_OPS_HPP

#include <type_traits>
#include <utility>

#include "config/common_type.hpp"
#include "config/details/binary_operators_helpers.hpp"
#include "config/type_traits.hpp"

#include "core/parameter_resolver.hpp"

#if MQ_HAS_CONCEPTS
#    include "config/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

namespace mindquantum {
namespace details::pr::traits {
template <typename type_t>
struct parameter_resolver_binop {
    using type = typename std::remove_cvref_t<type_t>::value_type;

    template <typename other_t>
    using new_type_t = ParameterResolver<other_t>;
};
}  // namespace details::pr::traits

// -----------------------------------------------------------------------------+

#define MQ_PR_TRAITS details::pr::traits::parameter_resolver_binop

#if MQ_HAS_CONCEPTS
#    define MQ_DEFINE_PR_BINOP_COMMUTATIVE(op, op_impl)                                                                \
        MQ_DEFINE_BINOP_SCALAR_LEFT_(op, op_impl, MQ_PR_TRAITS, concepts::parameter_resolver_scalar,                   \
                                     concepts::parameter_resolver)                                                     \
        MQ_DEFINE_BINOP_SCALAR_RIGHT_(op, op_impl, MQ_PR_TRAITS, concepts::parameter_resolver,                         \
                                      concepts::parameter_resolver_scalar)                                             \
        MQ_DEFINE_BINOP_TERMS_(op, op_impl, MQ_PR_TRAITS, concepts::parameter_resolver, concepts::parameter_resolver)
#    define MQ_DEFINE_PR_BINOP_NON_COMMUTATIVE(op, op_impl, op_inv)                                                    \
        MQ_DEFINE_BINOP_TERMS_(op, op_impl, MQ_PR_TRAITS, concepts::parameter_resolver, concepts::parameter_resolver)  \
        MQ_DEFINE_BINOP_SCALAR_RIGHT_(op, op_impl, MQ_PR_TRAITS, concepts::parameter_resolver,                         \
                                      concepts::parameter_resolver_scalar)                                             \
        template <concepts::parameter_resolver_scalar scalar_t, concepts::parameter_resolver rhs_t>                    \
        auto operator op(scalar_t&& lhs, rhs_t&& rhs) {                                                                \
            return (op_inv);                                                                                           \
        }
#else
namespace details::pr::traits {
namespace mq_traits = mindquantum::traits;

template <typename lhs_t, typename rhs_t>
inline constexpr auto lhs_and_scalar_v
    = mq_traits::is_parameter_resolver_decay_v<lhs_t>&& mq_traits::is_scalar_decay_v<rhs_t>;

template <typename lhs_t, typename rhs_t>
inline constexpr auto scalar_and_rhs_v
    = mq_traits::is_scalar_decay_v<lhs_t>&& mq_traits::is_parameter_resolver_decay_v<rhs_t>;

template <typename lhs_t, typename rhs_t>
inline constexpr auto is_compatible_v
    = (mq_traits::is_parameter_resolver_decay_v<lhs_t> && mq_traits::is_parameter_resolver_decay_v<rhs_t>)
      || lhs_and_scalar_v<lhs_t, rhs_t> || scalar_and_rhs_v<lhs_t, rhs_t>;
}  // namespace details::pr::traits

#    define MQ_DEFINE_PR_BINOP_COMMUTATIVE(op, op_impl)                                                                \
        MQ_DEFINE_BINOP_COMMUTATIVE_IMPL(op, op_impl, MQ_PR_TRAITS, details::pr::traits::is_compatible_v,              \
                                         traits::is_parameter_resolver_decay_v, traits::is_parameter_resolver_decay_v)
#    define MQ_DEFINE_PR_BINOP_NON_COMMUTATIVE(op, op_impl, op_inv)                                                    \
        MQ_DEFINE_BINOP_NON_COMMUTATIVE_IMPL(op, op_impl, op_inv, MQ_PR_TRAITS, details::pr::traits::is_compatible_v,  \
                                             traits::is_parameter_resolver_decay_v,                                    \
                                             traits::is_parameter_resolver_decay_v)
#endif  // MQ_HAS_CONCEPTS

MQ_DEFINE_PR_BINOP_COMMUTATIVE(+, config::details::plus_equal)
MQ_DEFINE_PR_BINOP_COMMUTATIVE(*, config::details::multiplies_equal)
MQ_DEFINE_PR_BINOP_NON_COMMUTATIVE(-, config::details::minus_equal, (-rhs + lhs))
MQ_DEFINE_PR_BINOP_COMMUTATIVE(/, config::details::divides_equal)

#undef MQ_PR_TRAITS
#undef MQ_DEFINE_PR_BINOP_COMMUTATIVE
#undef MQ_DEFINE_PR_BINOP_NON_COMMUTATIVE
#undef MQ_DEFINE_PR_BINOP_SCALAR_RIGHT_ONLY

// =============================================================================

#if MQ_HAS_CONCEPTS
template <concepts::parameter_resolver lhs_t, concepts::parameter_resolver rhs_t>
bool operator==(const lhs_t& lhs, const rhs_t& rhs) {
    return lhs.IsEqual(rhs);
}
template <concepts::parameter_resolver lhs_t, concepts::parameter_resolver_scalar rhs_t>
bool operator==(const lhs_t& lhs, const rhs_t& rhs) {
    return lhs.IsEqual(rhs);
}
template <concepts::parameter_resolver_scalar lhs_t, concepts::parameter_resolver rhs_t>
bool operator==(const lhs_t& lhs, const rhs_t& rhs) {
    return rhs.IsEqual(lhs);
}
#else
template <typename lhs_t, typename rhs_t,
          typename = std::enable_if_t<(traits::is_parameter_resolver_v<lhs_t> && traits::is_parameter_resolver_v<rhs_t>)
                                      || (traits::is_parameter_resolver_v<lhs_t> && traits::is_scalar_decay_v<rhs_t>)
                                      || (traits::is_scalar_decay_v<lhs_t> && traits::is_parameter_resolver_v<rhs_t>)>>
bool operator==(const lhs_t& lhs, const rhs_t& rhs) {
    return lhs.IsEqual(rhs);
}
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

#if !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
#    if MQ_HAS_CONCEPTS
template <concepts::parameter_resolver lhs_t, concepts::parameter_resolver rhs_t>
auto operator!=(const lhs_t& lhs, const rhs_t& rhs) {
    return !lhs.IsEqual(rhs);
}
template <concepts::parameter_resolver lhs_t, concepts::parameter_resolver_scalar rhs_t>
auto operator!=(const lhs_t& lhs, const rhs_t& rhs) {
    return !lhs.IsEqual(rhs);
}
template <concepts::parameter_resolver_scalar lhs_t, concepts::parameter_resolver rhs_t>
auto operator!=(const lhs_t& lhs, const rhs_t& rhs) {
    return !rhs.IsEqual(lhs);
}
#    else
template <typename lhs_t, typename rhs_t,
          typename = std::enable_if_t<(traits::is_parameter_resolver_v<lhs_t> && traits::is_parameter_resolver_v<rhs_t>)
                                      || (traits::is_parameter_resolver_v<lhs_t> && traits::is_scalar_decay_v<rhs_t>)
                                      || (traits::is_scalar_decay_v<lhs_t> && traits::is_parameter_resolver_v<rhs_t>)>>
auto operator!=(const lhs_t& lhs, const rhs_t& rhs) {
    return !lhs.IsEqual(rhs);
}
#    endif  // MQ_HAS_CONCEPTS
#endif      // !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
}  // namespace mindquantum

#endif /* PARAMETER_RESOLVER_EXTERNAL_OPS_HPP */
