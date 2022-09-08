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

#ifndef TERMS_OPERATOR_BASE_EXTERNAL_OPS_HPP
#define TERMS_OPERATOR_BASE_EXTERNAL_OPS_HPP

#include <type_traits>
#include <utility>

#include "config/type_traits.hpp"

#include "experimental/core/config.hpp"
#include "experimental/core/traits.hpp"
#include "experimental/ops/gates/terms_operator_base.hpp"
#include "experimental/ops/gates/traits.hpp"

#if MQ_HAS_CONCEPTS
#    include "config/concepts.hpp"

#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

namespace mindquantum::ops {
namespace details {
struct plus_equal {
    template <typename lhs_t, typename rhs_t>
    static constexpr auto apply(lhs_t& lhs, rhs_t&& rhs) {
        return lhs += std::forward<rhs_t>(rhs);
    }
};
struct minus_equal {
    template <typename lhs_t, typename rhs_t>
    static constexpr auto apply(lhs_t& lhs, rhs_t&& rhs) {
        return lhs -= std::forward<rhs_t>(rhs);
    }
};
struct multiplies_equal {
    template <typename lhs_t, typename rhs_t>
    static constexpr auto apply(lhs_t& lhs, rhs_t&& rhs) {
        return lhs *= std::forward<rhs_t>(rhs);
    }
};

struct divides_equal {
    template <typename lhs_t, typename rhs_t>
    static constexpr auto apply(lhs_t& lhs, rhs_t&& rhs) {
        return lhs /= std::forward<rhs_t>(rhs);
    }
};

template <typename func_t, typename terms_op_t, typename rhs_t>
auto r_value_optimisation(terms_op_t&& terms_op, rhs_t&& rhs) {
    if constexpr (std::is_same_v<terms_op_t, std::remove_cvref_t<terms_op_t>>) {
        // If terms_op is an r-value, we can safely modify and return it instead of creating a temporary
        func_t::apply(terms_op, std::forward<rhs_t>(rhs));
        return std::forward<terms_op_t>(terms_op);
    } else {
        std::remove_cvref_t<terms_op_t> tmp{terms_op};
        func_t::apply(tmp, std::forward<rhs_t>(rhs));
        return tmp;
    }
}

template <typename func_t, typename lhs_t, typename rhs_t>
auto terms_op_arithmetic_op_impl(lhs_t&& lhs, rhs_t&& rhs) {
    using left_coeff_t = typename std::remove_cvref_t<lhs_t>::coefficient_t;
    using right_coeff_t = typename std::remove_cvref_t<rhs_t>::coefficient_t;
    using common_t = std::common_type_t<left_coeff_t, right_coeff_t>;

    // See which of LHS or RHS we need to promote
    if constexpr (std::is_same_v<left_coeff_t, common_t>) {
        return r_value_optimisation<func_t>(std::forward<lhs_t>(lhs), std::forward<rhs_t>(rhs));
    } else {
        return r_value_optimisation<func_t>(std::forward<rhs_t>(rhs), std::forward<lhs_t>(lhs));
    }
}

template <typename func_t, typename term_op_t, typename scalar_t>
auto terms_op_arithmetic_scalar_op_impl(term_op_t&& term_op, scalar_t&& scalar) {
    static_assert(traits::is_terms_operator_v<term_op_t>);
    static_assert(!traits::is_terms_operator_v<scalar_t>);

    using terms_op_t = std::remove_cvref_t<term_op_t>;
    using common_t = std::common_type_t<typename terms_op_t::coefficient_t, std::remove_cvref_t<scalar_t>>;
    if constexpr (std::is_same_v<common_t, typename terms_op_t::coefficient_t>) {
        return r_value_optimisation<func_t>(std::forward<term_op_t>(term_op), std::forward<scalar_t>(scalar));
    } else {
        return r_value_optimisation<func_t>(typename terms_op_t::template new_derived_t<common_t>{term_op},
                                            std::forward<scalar_t>(scalar));
    }
}
}  // namespace details

// -----------------------------------------------------------------------------+

#define MQ_BINOP_TERMS_OP_IMPL_(op_impl, lhs, rhs)                                                                     \
    details::op_impl(std::forward<lhs##_t>(lhs), std::forward<rhs##_t>(rhs))
#if MQ_HAS_CONCEPTS
#    define MQ_DEFINE_BINOP_TERM_OPS_SCALAR_LEFT_(op, op_impl)                                                         \
        template <concepts::not_terms_op scalar_t, concepts::terms_op rhs_t>                                           \
        auto operator op(scalar_t&& scalar, rhs_t&& rhs) {                                                             \
            return MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, rhs, scalar);                  \
        }
#    define MQ_DEFINE_BINOP_TERM_OPS_SCALAR_RIGHT_(op, op_impl)                                                        \
        template <concepts::terms_op lhs_t, concepts::not_terms_op scalar_t>                                           \
        auto operator op(lhs_t&& lhs, scalar_t&& scalar) {                                                             \
            return MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, lhs, scalar);                  \
        }
#    define MQ_DEFINE_BINOP_TERM_OPS_TERMS_(op, op_impl)                                                               \
        template <concepts::terms_op lhs_t, concepts::terms_op rhs_t>                                                  \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            return MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_op_impl<op_impl>, lhs, rhs);                            \
        }

#    define MQ_DEFINE_BINOP_COMMUTATIVE(op, op_impl)                                                                   \
        MQ_DEFINE_BINOP_TERM_OPS_SCALAR_LEFT_(op, op_impl)                                                             \
        MQ_DEFINE_BINOP_TERM_OPS_SCALAR_RIGHT_(op, op_impl)                                                            \
        MQ_DEFINE_BINOP_TERM_OPS_TERMS_(op, op_impl)
#    define MQ_DEFINE_BINOP_NON_COMMUTATIVE(op, op_impl, op_inv)                                                       \
        MQ_DEFINE_BINOP_TERM_OPS_TERMS_(op, op_impl)                                                                   \
        MQ_DEFINE_BINOP_TERM_OPS_SCALAR_RIGHT_(op, op_impl)                                                            \
        template <concepts::not_terms_op scalar_t, concepts::terms_op rhs_t>                                           \
        auto operator op(scalar_t&& lhs, rhs_t&& rhs) requires(concepts::compat_terms_op_scalar<scalar_t, rhs_t>) {    \
            return (op_inv);                                                                                           \
        }
#    define MQ_DEFINE_BINOP_SCALAR_RIGHT_ONLY(op, op_impl) MQ_DEFINE_BINOP_TERM_OPS_SCALAR_RIGHT_(op, op_impl)

#else
template <typename lhs_t, typename rhs_t, typename = void>
struct lhs_and_scalar : std::false_type {};
template <typename lhs_t, typename rhs_t>
struct lhs_and_scalar<lhs_t, rhs_t,
                      std::enable_if_t<traits::is_terms_operator_v<lhs_t> && !traits::is_terms_operator_v<rhs_t>>>
    : std::true_type {};

template <typename lhs_t, typename rhs_t, typename = void>
struct scalar_and_rhs : std::false_type {};
template <typename lhs_t, typename rhs_t>
struct scalar_and_rhs<lhs_t, rhs_t,
                      std::enable_if_t<!traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>>>
    : std::true_type {};

#    define MQ_BINOP_COMPLETE_IMPL_(terms_op, lhs_terms_op, rhs_terms_op)                                              \
        if constexpr (traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>) {                      \
            return terms_op;                                                                                           \
        } else if constexpr (traits::is_terms_operator_v<lhs_t>) {                                                     \
            return lhs_terms_op;                                                                                       \
        } else {                                                                                                       \
            return rhs_terms_op;                                                                                       \
        }

#    define MQ_DEFINE_BINOP_COMMUTATIVE(op, op_impl)                                                                   \
        template <typename lhs_t, typename rhs_t,                                                                      \
                  typename                                                                                             \
                  = std::enable_if_t<(traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>)        \
                                     || lhs_and_scalar<lhs_t, rhs_t>::value || scalar_and_rhs<lhs_t, rhs_t>::value>>   \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            MQ_BINOP_COMPLETE_IMPL_((MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_op_impl<op_impl>, lhs, rhs)),         \
                                    (MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, lhs, rhs)),  \
                                    (MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, rhs, lhs)))  \
        }
#    define MQ_DEFINE_BINOP_NON_COMMUTATIVE(op, op_impl, op_inv)                                                       \
        template <typename lhs_t, typename rhs_t,                                                                      \
                  typename                                                                                             \
                  = std::enable_if_t<(traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>)        \
                                     || lhs_and_scalar<lhs_t, rhs_t>::value || scalar_and_rhs<lhs_t, rhs_t>::value>>   \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            MQ_BINOP_COMPLETE_IMPL_((MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_op_impl<op_impl>, lhs, rhs)),         \
                                    (MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, lhs, rhs)),  \
                                    (op_inv))                                                                          \
        }
#    define MQ_DEFINE_BINOP_SCALAR_RIGHT_ONLY(op, op_impl)                                                             \
        template <typename lhs_t, typename rhs_t, typename = std::enable_if_t<lhs_and_scalar<lhs_t, rhs_t>::value>>    \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            return MQ_BINOP_TERMS_OP_IMPL_(terms_op_arithmetic_scalar_op_impl<op_impl>, lhs, rhs);                     \
        }
#endif  // MQ_HAS_CONCEPTS

MQ_DEFINE_BINOP_COMMUTATIVE(+, details::plus_equal)
MQ_DEFINE_BINOP_COMMUTATIVE(*, details::multiplies_equal)
MQ_DEFINE_BINOP_NON_COMMUTATIVE(-, details::minus_equal, (-rhs + lhs))
MQ_DEFINE_BINOP_SCALAR_RIGHT_ONLY(/, details::divides_equal)

#undef MQ_DEFINE_BINOP_COMMUTATIVE
#undef MQ_DEFINE_BINOP_NON_COMMUTATIVE
#undef MQ_DEFINE_BINOP_SCALAR_RIGHT_ONLY
#undef MQ_BINOP_TERMS_OP_IMPL_

// =============================================================================

#if MQ_HAS_CONCEPTS
template <concepts::terms_op lhs_t, concepts::terms_op rhs_t>
#else
template <typename lhs_t, typename rhs_t,
          typename = std::enable_if_t<(traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>)>>
#endif  // MQ_HAS_CONCEPTS
auto operator==(const lhs_t& lhs, const rhs_t& rhs) {
    return lhs.is_equal(rhs);
}

#if !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
#    if MQ_HAS_CONCEPTS
template <concepts::terms_op lhs_t, concepts::terms_op rhs_t>
#    else
template <typename lhs_t, typename rhs_t,
          typename = std::enable_if_t<(traits::is_terms_operator_v<lhs_t> && traits::is_terms_operator_v<rhs_t>)>>
#    endif  // MQ_HAS_CONCEPTS
auto operator!=(const lhs_t& lhs, const rhs_t& rhs) {
    return !lhs.is_equal(rhs);
}
#endif  // !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
}  // namespace mindquantum::ops

#endif /* TERMS_OPERATOR_BASE_EXTERNAL_OPS_HPP */
