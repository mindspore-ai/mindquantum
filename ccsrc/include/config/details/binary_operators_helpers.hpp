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

#ifndef MQ_CONFIG_BINARY_OPERATORS_HELPERS_HPP
#define MQ_CONFIG_BINARY_OPERATORS_HELPERS_HPP

#include <type_traits>
#include <utility>

#include "config/common_type.hpp"
#include "config/config.hpp"

namespace mindquantum::config::details {
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

template <typename func_t, typename lhs_t, typename rhs_t>
auto r_value_optimisation(lhs_t&& lhs, rhs_t&& rhs) {
    if constexpr (std::is_same_v<lhs_t, std::remove_cvref_t<lhs_t>>) {
        // If terms_op is an r-value, we can safely modify and return it instead of creating a temporary
        func_t::apply(lhs, std::forward<rhs_t>(rhs));
        return std::forward<lhs_t>(lhs);
    } else {
        std::remove_cvref_t<lhs_t> tmp{lhs};
        func_t::apply(tmp, std::forward<rhs_t>(rhs));
        return tmp;
    }
}

//! Helper function to define an inplace arithmetic operator
/*!
 * Both LHS and RHS types will be passed through trait<T>::type in order to compute the common type as:
 * \c common_type<trait_t<lhs_t>::type, trait_t<rhs_t>::type>.
 *
 * If both LHS and RHS need to be converted, \c trait<T>::new_type_t<common_t> will be computed for both LHS and RHS.
 * Both of these need to result in the same type.
 *
 * \tparam func_t Type of in-place operator to use (e.g. \c mindquantum::config::plus_equal)
 * \tparam trait_t Type trait used to calculate the common type (if any) between LHS and RHS
 * \param lhs LHS of the in-place binary operator
 * \param rhs RHS of the in-place binary operator
 * \return \c lhs \c += \c rhs, the type of which might be \c lhs_t, \c rhs_t or \c trait_t::new_type<common_t>
 *
 * \note You might want to ensure that std::remove_cvref_t<T> is used within the \c trait_t template class
 */
template <typename func_t, template <typename type_t> typename trait_t, typename lhs_t, typename rhs_t>
auto arithmetic_op_impl(lhs_t&& lhs, rhs_t&& rhs) {
    using left_t = typename trait_t<lhs_t>::type;
    using right_t = typename trait_t<rhs_t>::type;
    using common_t = mindquantum::traits::common_type_t<left_t, right_t>;

    // See which of LHS or RHS we need to promote
    if constexpr (std::is_same_v<left_t, common_t>) {
        return config::details::r_value_optimisation<func_t>(std::forward<lhs_t>(lhs), std::forward<rhs_t>(rhs));
    } else if constexpr (std::is_same_v<right_t, common_t>) {
        return config::details::r_value_optimisation<func_t>(std::forward<rhs_t>(rhs), std::forward<lhs_t>(lhs));
    } else {
        /*
         * In this case, we need to convert both LHS and RHS (e.g. T<std::complex<double>> T<ParameterResolver<double>>)
         * -> make sure that both LHS and RHS would lead to same type  (e.g. both FermionOperator)
         */
        using lhs_new_t = typename trait_t<lhs_t>::template new_type_t<common_t>;
        using rhs_new_t = typename trait_t<rhs_t>::template new_type_t<common_t>;
        static_assert(std::is_same_v<lhs_new_t, rhs_new_t>);
        lhs_new_t tmp{lhs};
        func_t::apply(tmp, std::forward<rhs_t>(rhs));
        return tmp;
    }
}

//! Helper function to define an inplace arithmetic operator
/*!
 * Only the LHS type will be passed through trait<T>::type in order to compute the common type, which is then calculated
 * as:  \c common_type<trait_t<lhs_t>::type, rhs_t>.
 *
 * If both LHS and RHS need to be converted, \c trait<T>::new_type_t<common_t> will be computed for both LHS and RHS.
 * Both of these need to result in the same type.
 *
 * \tparam func_t Type of in-place operator to use (e.g. \c mindquantum::config::plus_equal)
 * \tparam trait_t Type trait used to calculate the common type (if any) between LHS and RHS
 * \param lhs LHS of the in-place binary operator
 * \param rhs RHS of the in-place binary operator
 * \return \c lhs \c += \c rhs, the type of which might be \c lhs_t, \c rhs_t or \c trait_t::new_type<common_t>
 */
template <typename func_t, template <typename type_t> typename trait_t, typename lhs_t, typename scalar_t>
auto arithmetic_scalar_op_impl(lhs_t&& lhs, scalar_t&& scalar) {
    using terms_op_t = std::remove_cvref_t<lhs_t>;
    using common_t
        = mindquantum::traits::common_type_t<typename trait_t<terms_op_t>::type, std::remove_cvref_t<scalar_t>>;
    if constexpr (std::is_same_v<common_t, typename trait_t<terms_op_t>::type>) {
        return config::details ::r_value_optimisation<func_t>(std::forward<lhs_t>(lhs), std::forward<scalar_t>(scalar));
    } else {
        return config::details ::r_value_optimisation<func_t>(
            typename trait_t<terms_op_t>::template new_type_t<common_t>{lhs}, std::forward<scalar_t>(scalar));
    }
}
}  // namespace mindquantum::config::details

// =============================================================================

#define MQ_BINOP_COMMA ,
#define MQ_BINOP_IMPL_(op_impl, op_impl_t, lhs, rhs)                                                                   \
    config::details::op_impl<op_impl_t>(std::forward<lhs##_t>(lhs), std::forward<rhs##_t>(rhs))

// -----------------------------------------------------------------------------

#if MQ_HAS_CONCEPTS
#    define MQ_DEFINE_BINOP_SCALAR_LEFT_(op, op_impl, traits_t, lhs_concept, rhs_concept)                              \
        template <lhs_concept scalar_t, rhs_concept rhs_t>                                                             \
        auto operator op(scalar_t&& scalar, rhs_t&& rhs) {                                                             \
            return MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, rhs, scalar);            \
        }
#    define MQ_DEFINE_BINOP_SCALAR_RIGHT_(op, op_impl, traits_t, lhs_concept, rhs_concept)                             \
        template <lhs_concept lhs_t, rhs_concept scalar_t>                                                             \
        auto operator op(lhs_t&& lhs, scalar_t&& scalar) {                                                             \
            return MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, scalar);            \
        }
#    define MQ_DEFINE_BINOP_TERMS_(op, op_impl, traits_t, lhs_concept, rhs_concept)                                    \
        template <lhs_concept lhs_t, rhs_concept rhs_t>                                                                \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            return MQ_BINOP_IMPL_(arithmetic_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, rhs);                      \
        }
#else
#    define MQ_BINOP_COMPLETE_IMPL_(terms_op, lhs_terms_op, rhs_terms_op, lhs_concept, rhs_concept)                    \
        static_assert(lhs_concept || rhs_concept);                                                                     \
        if constexpr (lhs_concept && rhs_concept) {                                                                    \
            return terms_op;                                                                                           \
        } else if constexpr (lhs_concept) {                                                                            \
            return lhs_terms_op;                                                                                       \
        } else {                                                                                                       \
            return rhs_terms_op;                                                                                       \
        }
#    define MQ_DEFINE_BINOP_COMMUTATIVE_IMPL(op, op_impl, traits_t, enabling_traits_v, lhs_concept, rhs_concept)       \
        template <typename lhs_t, typename rhs_t, typename = std::enable_if_t<enabling_traits_v<lhs_t, rhs_t>>>        \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            MQ_BINOP_COMPLETE_IMPL_(                                                                                   \
                (MQ_BINOP_IMPL_(arithmetic_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, rhs)),                       \
                (MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, rhs)),                \
                (MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, rhs, lhs)),                \
                lhs_concept<lhs_t>, rhs_concept<rhs_t>)                                                                \
        }
#    define MQ_DEFINE_BINOP_NON_COMMUTATIVE_IMPL(op, op_impl, op_inv, traits_t, enabling_traits_v, lhs_concept,        \
                                                 rhs_concept)                                                          \
        template <typename lhs_t, typename rhs_t, typename = std::enable_if_t<enabling_traits_v<lhs_t, rhs_t>>>        \
        auto operator op(lhs_t&& lhs, rhs_t&& rhs) {                                                                   \
            MQ_BINOP_COMPLETE_IMPL_(                                                                                   \
                (MQ_BINOP_IMPL_(arithmetic_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, rhs)),                       \
                (MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, rhs)), (op_inv),      \
                lhs_concept<lhs_t>, rhs_concept<rhs_t>)                                                                \
        }
#    define MQ_DEFINE_BINOP_SCALAR_RIGHT_ONLY_IMPL(op, op_impl, traits_t, enabling_traits_v)                           \
        template <typename lhs_t, typename scalar_t, typename = std::enable_if_t<enabling_traits_v<lhs_t, scalar_t>>>  \
        auto operator op(lhs_t&& lhs, scalar_t&& scalar) {                                                             \
            return MQ_BINOP_IMPL_(arithmetic_scalar_op_impl, op_impl MQ_BINOP_COMMA traits_t, lhs, scalar);            \
        }
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

#endif /* MQ_CONFIG_BINARY_OPERATORS_HELPERS_HPP */
