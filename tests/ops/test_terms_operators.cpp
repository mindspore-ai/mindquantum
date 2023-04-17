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

#include <algorithm>
#include <complex>
#include <functional>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <boost/range/combine.hpp>

#include "config/detected.hpp"
#include "config/logging.hpp"
#include "config/type_traits.hpp"

#include "math/pr/parameter_resolver.hpp"
#include "mindquantum/catch2/mindquantum.hpp"
#include "ops/gates/details/floating_point_coeff_policy.hpp"
#include "ops/gates/details/parameter_resolver_coeff_policy.hpp"
#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/terms_operator_base.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

using namespace std::literals::complex_literals;
using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;
using TermValue = mindquantum::ops::TermValue;

// =============================================================================

namespace order = mindquantum::ops::order;
using mindquantum::ops::py_terms_t;
using mindquantum::ops::term_t;
using mindquantum::ops::terms_t;

namespace {
template <typename coefficient_t>
struct DummyOperatorTermPolicy {
    static auto to_string(const mindquantum::ops::TermValue& value) {
        if (value == TermValue::X) {
            return "X"s;
        }
        if (value == TermValue::Y) {
            return "Y"s;
        }
        if (value == TermValue::Z) {
            return "Z"s;
        }
        if (value == TermValue::a) {
            return "v"s;
        }
        if (value == TermValue::adg) {
            return "^"s;
        }
        return "UNKNOWN"s;
    }
    static auto to_string(const mindquantum::ops::term_t& term) -> std::string {
        return fmt::format("{}-{}", std::get<0>(term), to_string(std::get<1>(term)));
    }

    static auto parse_terms_string(std::string_view /* terms_string */) -> terms_t {
        return {};
    }

    static std::tuple<std::vector<term_t>, coefficient_t> simplify(terms_t terms, coefficient_t coeff = 1.) {
        return {std::move(terms), coeff};
    }

    static std::tuple<std::vector<term_t>, coefficient_t> simplify(py_terms_t py_terms, coefficient_t coeff = 1.) {
        return {std::move(py_terms), coeff};
    }

    static std::pair<terms_t, coefficient_t> sort_terms(terms_t local_ops, coefficient_t coeff) {
        return {std::move(local_ops), coeff};
    }
};

template <typename coeff_t>
struct DummyOperator : mindquantum::ops::TermsOperatorBase<DummyOperator, coeff_t, DummyOperatorTermPolicy> {
    using base_t = mindquantum::ops::TermsOperatorBase<DummyOperator, coeff_t, DummyOperatorTermPolicy>;
    using mindquantum::ops::TermsOperatorBase<DummyOperator, coeff_t, DummyOperatorTermPolicy>::TermsOperatorBase;

    DummyOperator(const DummyOperator&) = default;
    DummyOperator(DummyOperator&&) = default;
    DummyOperator& operator=(const DummyOperator&) = default;
    DummyOperator& operator=(DummyOperator&&) = default;
    ~DummyOperator() = default;

    using term_t = mindquantum::ops::term_t;
    using py_term_t = mindquantum::ops::py_term_t;
};
}  // namespace

using DummyOperatorD = DummyOperator<double>;
using DummyOperatorCD = DummyOperator<std::complex<double>>;
using coefficient_t = DummyOperatorCD::coefficient_t;
using coeff_term_dict_t = DummyOperatorCD::coeff_term_dict_t;

// =============================================================================
// =============================================================================
// Static (ie. compile-time) testing

#if !((defined __GNUC__) && __GNUC__ == 7 && __GNUC_MINOR__ < 5)
static_assert(std::size(DummyOperatorCD::kind()) >= 35 /* = len(DummyOperator<std::complex<double>>) */);
#endif  // GCC < 7.5

namespace {
#if MQ_HAS_CONCEPTS
template <typename coeff_t>
struct A {
    using terms_operator_tag = void;
    using coefficient_t = coeff_t;
    using coefficient_real_t = typename mindquantum::traits::to_real_type_t<coefficient_t>;
    static constexpr auto is_real_valued = std::is_same_v<coefficient_t, coefficient_real_t>;
};
template <typename coeff_t>
struct B : A<coeff_t> {};

static_assert(mindquantum::concepts::compat_terms_op<B<double>, A<double>>);
static_assert(!mindquantum::concepts::compat_terms_op<B<std::complex<double>>, A<double>>);
static_assert(mindquantum::concepts::compat_terms_op<B<double>, A<std::complex<double>>>);
static_assert(mindquantum::concepts::compat_terms_op<B<std::complex<double>>, A<std::complex<double>>>);

static_assert(mindquantum::concepts::compat_terms_op_scalar<double, A<double>>);
static_assert(!mindquantum::concepts::compat_terms_op_scalar<std::complex<double>, A<double>>);
static_assert(mindquantum::concepts::compat_terms_op_scalar<double, A<std::complex<double>>>);
static_assert(mindquantum::concepts::compat_terms_op_scalar<std::complex<double>, A<std::complex<double>>>);
#endif  // MQ_HAS_CONCEPTS

// -----------------------------------------------------------------------------

template <typename lhs_t, typename rhs_t, typename binop_t>
using binop_valid = decltype(binop_t{}(std::declval<lhs_t>(), std::declval<rhs_t>()));  // NOLINT(whitespace/braces)

template <typename lhs_t, typename rhs_t, typename binop_t>
inline constexpr auto is_binop_defined_v = mindquantum::is_detected_v<binop_valid, lhs_t, rhs_t, binop_t>;

// -------------------------------------

template <typename lhs_t, typename rhs_t>
using iadd_valid = decltype(std::declval<lhs_t&>() += std::declval<rhs_t>());
template <typename lhs_t, typename rhs_t>
using isub_valid = decltype(std::declval<lhs_t&>() -= std::declval<rhs_t>());
template <typename lhs_t, typename rhs_t>
using imul_valid = decltype(std::declval<lhs_t&>() *= std::declval<rhs_t>());
template <typename lhs_t, typename rhs_t>
using idiv_valid = decltype(std::declval<lhs_t&>() /= std::declval<rhs_t>());

template <typename lhs_t, typename rhs_t, typename unop_t>
using inplace_unop_valid = decltype(unop_t::apply(std::declval<lhs_t&>(), std::declval<rhs_t>()));

template <typename lhs_t, typename rhs_t, template <typename olhs_t, typename orhs_t> typename unop_t>
inline constexpr auto is_unop_defined_v = mindquantum::is_detected_v<unop_t, lhs_t, rhs_t>;

// -------------------------------------

template <typename lhs_t, typename rhs_t, typename binop_t>
using binop_ret_t = mindquantum::detected_t<binop_valid, lhs_t, rhs_t, binop_t>;

// -----------------------------------------------------------------------------

// clang-format off
#define MQ_IS_BINOP_DEFINED(op1_t, op2_t)                                                                              \
    (is_binop_defined_v<op1_t, op2_t, std::plus<>>                                                                     \
    && is_binop_defined_v<op1_t, op2_t, std::minus<>>                                                                  \
    && is_binop_defined_v<op1_t, op2_t, std::multiplies<>>)
#define MQ_IS_BINOP_RETURN_TYPE(op1_t, op2_t, ret_t)                                                                   \
    (mindquantum::traits::is_same_decay_v<binop_ret_t<op1_t, op2_t, std::plus<>>, ret_t>                               \
    && mindquantum::traits::is_same_decay_v<binop_ret_t<op1_t, op2_t, std::minus<>>, ret_t>                            \
    && mindquantum::traits::is_same_decay_v<binop_ret_t<op1_t, op2_t, std::multiplies<>>, ret_t>)

#define MQ_IS_INPLACE_UNOP_VALID(op1_t, op2_t)                                                                         \
    (is_unop_defined_v<op1_t, op2_t, iadd_valid>                                                                       \
    && is_unop_defined_v<op1_t, op2_t, isub_valid>                                                                     \
    && is_unop_defined_v<op1_t, op2_t, imul_valid>)
#define MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op1_t, op2_t)                                                                \
    (MQ_IS_INPLACE_UNOP_VALID(op1_t, op2_t) && is_unop_defined_v<op1_t, op2_t, idiv_valid>)

#define MQ_BINOP_COMMUTATIVE_VALID(op1_t, op2_t, ret_t)                                                                \
    (MQ_IS_BINOP_DEFINED(op1_t, op2_t) && MQ_IS_BINOP_DEFINED(op2_t, op1_t) \
     && MQ_IS_BINOP_RETURN_TYPE(op1_t, op2_t, ret_t) && MQ_IS_BINOP_RETURN_TYPE(op2_t, op1_t, ret_t))
#define MQ_BINOP_SCALAR_RIGHT_VALID(term_op_t, scalar_t, ret_t)                                                        \
    ((MQ_IS_BINOP_DEFINED(term_op_t, scalar_t) && (is_binop_defined_v<term_op_t, scalar_t, std::divides<>>))           \
    && (MQ_IS_BINOP_RETURN_TYPE(term_op_t, scalar_t, ret_t)                                                            \
        && mindquantum::traits::is_same_decay_v<binop_ret_t<term_op_t, scalar_t, std::divides<>>, ret_t>))
// clang-format on
#define MQ_BINOP_SCALAR_LEFT_VALID(term_op_t, scalar_t, ret_t)                                                         \
    (MQ_IS_BINOP_DEFINED(scalar_t, term_op_t) && MQ_IS_BINOP_RETURN_TYPE(scalar_t, term_op_t, ret_t))

template <typename coeff_t>
using op_t = DummyOperator<coeff_t>;

template <typename float_t>
using cmplx_t = std::complex<float_t>;

template <typename float_t>
using pr_t = mindquantum::ParameterResolver<float_t>;

// -----------------------------------------------------------------------------

// TermsOp - TermsOp inplace operators
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<float>, op_t<float>));
static_assert(!MQ_IS_INPLACE_UNOP_VALID(op_t<float>, op_t<std::complex<float>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<float>, op_t<double>));
static_assert(!MQ_IS_INPLACE_UNOP_VALID(op_t<float>, op_t<std::complex<double>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<double>, op_t<float>));
static_assert(!MQ_IS_INPLACE_UNOP_VALID(op_t<double>, op_t<std::complex<float>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<double>, op_t<double>));
static_assert(!MQ_IS_INPLACE_UNOP_VALID(op_t<double>, op_t<std::complex<double>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<float>>, op_t<float>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<float>>, op_t<std::complex<float>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<float>>, op_t<double>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<float>>, op_t<std::complex<double>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<double>>, op_t<float>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<double>>, op_t<std::complex<float>>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<double>>, op_t<double>));
static_assert(MQ_IS_INPLACE_UNOP_VALID(op_t<std::complex<double>>, op_t<std::complex<double>>));

// TermsOp - Scalar inplace operators

static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<float>, float));
static_assert(!MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<float>, std::complex<float>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<float>, double));
static_assert(!MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<float>, std::complex<double>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<double>, float));
static_assert(!MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<double>, std::complex<float>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<double>, double));
static_assert(!MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<double>, std::complex<double>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<float>>, float));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<float>>, std::complex<float>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<float>>, double));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<float>>, std::complex<double>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<double>>, float));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<double>>, std::complex<float>));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<double>>, double));
static_assert(MQ_IS_INPLACE_UNOP_WITH_DIV_VALID(op_t<std::complex<double>>, std::complex<double>));

// -----------------------------------------------------------------------------

// TermsOp - TermsOp binary operators
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<float>, op_t<float>, op_t<float>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<float>, op_t<double>, op_t<double>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<float>, op_t<cmplx_t<float>>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<float>, op_t<cmplx_t<double>>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<double>, op_t<double>, op_t<double>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<double>, op_t<cmplx_t<double>>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<cmplx_t<float>>, op_t<cmplx_t<float>>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<cmplx_t<float>>, op_t<cmplx_t<double>>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_COMMUTATIVE_VALID(op_t<cmplx_t<double>>, op_t<cmplx_t<double>>, op_t<cmplx_t<double>>));

// Scalar - TermsOp binary operators
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<float>, float, op_t<float>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<float>, double, op_t<double>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<float>, cmplx_t<float>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<float>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<double>, float, op_t<double>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<double>, double, op_t<double>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<double>, cmplx_t<float>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<double>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<float>>, float, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<float>>, double, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<float>>, cmplx_t<float>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<float>>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<float>>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<double>>, float, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<double>>, double, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<double>>, cmplx_t<float>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_LEFT_VALID(op_t<cmplx_t<double>>, cmplx_t<double>, op_t<cmplx_t<double>>));

// TermsOp - Scalar binary operators
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<float>, float, op_t<float>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<float>, double, op_t<double>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<float>, cmplx_t<float>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<float>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<double>, float, op_t<double>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<double>, double, op_t<double>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<double>, cmplx_t<float>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<double>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<float>>, float, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<float>>, double, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<float>>, cmplx_t<float>, op_t<cmplx_t<float>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<float>>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<float>>, cmplx_t<double>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<double>>, float, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<double>>, double, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<double>>, cmplx_t<float>, op_t<cmplx_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<cmplx_t<double>>, cmplx_t<double>, op_t<cmplx_t<double>>));

static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<float>>, float, op_t<pr_t<float>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<float>>, double, op_t<pr_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<float>>, cmplx_t<float>, op_t<pr_t<cmplx_t<float>>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<float>>, cmplx_t<double>, op_t<pr_t<cmplx_t<double>>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<float>>, cmplx_t<double>, op_t<pr_t<cmplx_t<double>>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<double>>, float, op_t<pr_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<double>>, double, op_t<pr_t<double>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<double>>, cmplx_t<float>, op_t<pr_t<cmplx_t<double>>>));
static_assert(MQ_BINOP_SCALAR_RIGHT_VALID(op_t<pr_t<double>>, cmplx_t<double>, op_t<pr_t<cmplx_t<double>>>));
}  // namespace

// =============================================================================
// =============================================================================

TEST_CASE("TermsOperator constructor", "[terms_op][ops]") {
    coeff_term_dict_t ref_terms;

    SECTION("Default constructor") {
        DummyOperatorCD op;
        CHECK(std::empty(op));
        CHECK(std::size(op) == 0);
        CHECK(std::empty(op.get_terms()));
        CHECK(std::size(op.get_terms()) == 0);
        CHECK(op.num_targets() == 0);
        CHECK(op.count_qubits() == 0);
        CHECK(op.is_identity());
        CHECK(!op.is_singlet());
        CHECK(std::empty(op.singlet()));
    }
    SECTION("Identity") {
        DummyOperatorCD identity{terms_t{}};
        CHECK(!std::empty(identity));
        CHECK(std::size(identity) == 1);
        CHECK(!std::empty(identity.get_terms()));
        CHECK(std::size(identity.get_terms()) == 1);
        CHECK(identity.num_targets() == 1);
        CHECK(identity.count_qubits() == 1);
        CHECK(identity.is_identity());
        CHECK(identity.is_singlet());
    }
    SECTION("Identity (static member function)") {
        auto identity = DummyOperatorCD::identity();
        CHECK(!std::empty(identity));
        CHECK(std::size(identity) == 1);
        CHECK(!std::empty(identity.get_terms()));
        CHECK(std::size(identity.get_terms()) == 1);
        CHECK(identity.num_targets() == 1);
        CHECK(identity.count_qubits() == 1);
        CHECK(identity.is_identity());
        CHECK(identity.is_singlet());
    }
    SECTION("Single term constructor") {
        const auto ref_term = term_t{0, TermValue::X};
        ref_terms.emplace_back(terms_t{ref_term}, 1);

        DummyOperatorCD op{ref_term};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 1);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 1);
        CHECK(op.count_qubits() == 1);
        CHECK(!op.is_identity());
        CHECK(op.is_singlet());
    }
    SECTION("Single term multiple local_ops constructor") {
        const auto ref_term = terms_t{{0, TermValue::X}, {1, TermValue::X}};
        const auto coeff = 0.5;
        ref_terms.emplace_back(ref_term, coeff);

        DummyOperatorCD op{ref_term, coeff};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 1);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 2);
        CHECK(op.count_qubits() == 2);
        CHECK(!op.is_identity());
        CHECK(op.is_singlet());

        const auto singlet_terms = op.singlet();
        REQUIRE(std::size(ref_term) == std::size(singlet_terms));
        CHECK(coeff == op.singlet_coeff());

        for (const auto& [ref, test_op] : boost::combine(ref_term, singlet_terms)) {
            const auto& test_terms = test_op.get_terms();
            REQUIRE(std::size(test_terms) == 1);
            CHECK(begin(test_terms)->second == 1.);
            const auto& test = begin(test_terms)->first;
            REQUIRE(std::size(test) == 1);
            CHECK(ref == test.front());
        }
    }
    SECTION("Multiple terms constructor") {
        ref_terms.emplace_back(terms_t{{0, TermValue::Y}, {3, TermValue::Z}}, 0.75i);
        ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 2.);

        DummyOperatorCD op{ref_terms};
        CHECK(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(!std::empty(op.get_terms()));
        REQUIRE(std::size(op.get_terms()) == 2);
        CHECK(op.get_terms() == ref_terms);

        CHECK(op.num_targets() == 4);
        CHECK(op.count_qubits() == 4);
        CHECK(!op.is_identity());
        CHECK(!op.is_singlet());
    }
}

// =============================================================================

TEST_CASE("TermsOperator identity", "[terms_op][ops]") {
    SECTION("Empty term operator") {
        DummyOperatorCD op;
        CHECK(op.is_identity());

        CHECK(op.constant() == 0.);
        CHECK(std::empty(op));

        const auto coeff = 2. + 3.i;
        op.constant(coeff);  // Will insert an empty term {}
        CHECK(!std::empty(op));
        CHECK(op.constant() == 2. + 3.i);
        CHECK(op.is_singlet());
    }
    SECTION("Empty term operator") {
        auto const coeff = 1. + 2.i;
        DummyOperatorCD op{terms_t{}, coeff};
        CHECK(op.is_identity());
        CHECK(op.constant() == coeff);
        CHECK(op.is_singlet());
    }
    SECTION("Single term operator") {
        DummyOperatorCD op{term_t{0, TermValue::X}, 1.e-5};
        CHECK(!op.is_identity());
        CHECK(op.is_identity(1.e-5));
        CHECK(op.is_singlet());
    }
    SECTION("Single term multiple local_ops operator") {
        const auto terms = terms_t{{0, TermValue::X}, {2, TermValue::Z}};
        DummyOperatorCD op{terms, 1.e-4};
        CHECK(!op.is_identity());
        CHECK(op.is_identity(1.e-4));
        CHECK(op.is_singlet());

        const auto singlet_terms = op.singlet();
        REQUIRE(std::size(singlet_terms) == std::size(terms));
    }
    SECTION("Single term multiple local_ops operator") {
        auto terms = coeff_term_dict_t{};
        terms.emplace_back(terms_t{{0, TermValue::Y}}, 1.e-5);
        terms.emplace_back(terms_t{{2, TermValue::Z}, {4, TermValue::X}}, 1.e-6);
        DummyOperatorCD op{terms};
        CHECK(!op.is_identity());
        CHECK(!op.is_identity(1.e-6));
        CHECK(op.is_identity(1.e-5));
    }
    SECTION("Single term multiple local_ops including I operator") {
        auto terms = coeff_term_dict_t{};
        terms.emplace_back(terms_t{{0, TermValue::I}, {1, TermValue::I}}, 2i);
        terms.emplace_back(terms_t{{3, TermValue::X}}, DummyOperatorCD::EQ_TOLERANCE / 10.);
        terms.emplace_back(terms_t{{1, TermValue::I}}, 1);
        DummyOperatorCD op{terms};
        CHECK(op.is_identity());
        CHECK(!op.is_singlet());
    }
}

// =============================================================================

TEST_CASE("TermsOperator cast", "[terms_op][ops]") {
    auto terms = DummyOperatorD::coeff_term_dict_t{};
    terms.emplace_back(terms_t{{0, TermValue::Y}}, 1.);
    terms.emplace_back(terms_t{{2, TermValue::Z}, {4, TermValue::X}}, 3.2);

    DummyOperatorD real_op{terms};

    const auto cmplx_op = real_op.cast<DummyOperatorCD>();

    REQUIRE(std::size(cmplx_op) == std::size(real_op));
    for (const auto& [orig_terms, cmplx_terms] : boost::combine(real_op.get_terms(), cmplx_op.get_terms())) {
        CHECK(orig_terms.first == cmplx_terms.first);
        CHECK(static_cast<std::complex<double>>(orig_terms.second) == cmplx_terms.second);
    }

    const auto pr_op = real_op.cast<DummyOperator<mindquantum::ParameterResolver<double>>>();

    REQUIRE(std::size(pr_op) == std::size(real_op));
    for (const auto& [orig_terms, pr_terms] : boost::combine(real_op.get_terms(), pr_op.get_terms())) {
        CHECK(orig_terms.first == pr_terms.first);
        CHECK(pr_terms.second.IsConst());
        CHECK(orig_terms.second == pr_terms.second.const_value);
    }
}

TEST_CASE("TermsOperator real/imag cast", "[terms_op][ops]") {
    auto terms = coeff_term_dict_t{};
    terms.emplace_back(terms_t{{0, TermValue::Y}}, 1. + 2.i);
    terms.emplace_back(terms_t{{2, TermValue::Z}, {4, TermValue::X}}, 3. + 4.i);
    DummyOperatorCD op{terms};

    auto real = op.real();
    REQUIRE(std::size(real) == std::size(op));
    for (const auto& [orig_terms, real_terms] : boost::combine(op.get_terms(), real.get_terms())) {
        CHECK(orig_terms.first == real_terms.first);
        CHECK(std::real(orig_terms.second) == real_terms.second);
    }

    auto imag = op.imag();
    REQUIRE(std::size(imag) == std::size(op));
    for (const auto& [orig_terms, imag_terms] : boost::combine(op.get_terms(), imag.get_terms())) {
        CHECK(orig_terms.first == imag_terms.first);
        CHECK(std::imag(orig_terms.second) == imag_terms.second);
    }
}

// =============================================================================

TEST_CASE("TermsOperator compression", "[terms_op][ops]") {
    const auto coeff1 = 1.e-5;
    const auto terms1 = terms_t{{0, TermValue::Y}};
    const auto coeff2 = 1.e-6;
    const auto terms2 = terms_t{{2, TermValue::Z}, {4, TermValue::X}};

    auto terms = coeff_term_dict_t{};
    terms.emplace_back(terms1, coeff1);
    terms.emplace_back(terms2, coeff2);
    DummyOperatorCD op{terms};

    REQUIRE(std::size(op) == 2);
    op.compress(1.e-6);

    REQUIRE(std::size(op) == 1);
    REQUIRE(begin(op.get_terms())->first == terms1);
    REQUIRE(begin(op.get_terms())->second == coeff1);

    op.compress(1.e-4);
    REQUIRE(std::empty(op));
}

// =============================================================================

TEST_CASE("TermsOperator to_string", "[terms_op][ops]") {
    auto str = ""s;
    auto ref_str = ""s;

    SECTION("0") {
        str = DummyOperatorCD(terms_t{{0, TermValue::a}}).to_string();
        ref_str = "1 [0-v]";
    }
    SECTION("1^") {
        str = DummyOperatorCD(terms_t{{1, TermValue::adg}}).to_string();
        ref_str = "1 [1-^]";
    }
    SECTION("Identity") {
        str = DummyOperatorCD::identity().to_string();
        ref_str = "1 []";
    }
    SECTION("1v 2^ X3 Y4 Z5 + Identity") {
        str = DummyOperatorCD(
                  terms_t{
                      {1, TermValue::a}, {2, TermValue::adg}, {3, TermValue::X}, {4, TermValue::Y}, {5, TermValue::Z}},
                  1.2i)
                  .to_string();
        ref_str = "1.2j [1-v 2-^ 3-X 4-Y 5-Z]";
    }

    INFO("ref_str = " << ref_str);
    CHECK(ref_str == str);
}

TEST_CASE("DummyOperatorD dumps", "[terms_op][ops]") {
    DummyOperatorCD op;
    auto ref_json = ""s;
    SECTION("Empty") {
        op = DummyOperatorCD();
        ref_json = R"({
    "num_targets_": 0,
    "terms_": []
})"s;
    }
    SECTION("1v 2^ X3 Y4 Z5 + Identity") {
        op = DummyOperatorCD(
                 terms_t{
                     {1, TermValue::a}, {2, TermValue::adg}, {3, TermValue::X}, {4, TermValue::Y}, {5, TermValue::Z}},
                 1.2i)
             + DummyOperatorCD::identity();
        const auto num_targets = op.num_targets();
        ref_json = R"s({
    "num_targets_": 6,
    "terms_": [
        [
            [
                [
                    1,
                    "v"
                ],
                [
                    2,
                    "^"
                ],
                [
                    3,
                    "X"
                ],
                [
                    4,
                    "Y"
                ],
                [
                    5,
                    "Z"
                ]
            ],
            [
                0.0,
                1.2
            ]
        ],
        [
            [],
            [
                1.0,
                0.0
            ]
        ]
    ]
})s";
    }

    CHECK(op.dumps() == ref_json);
}

// TEST_CASE("FermionOperator loads", "[terms_op][ops]") {
//     std::string json_data;
//     std::optional<FermionOperator> fermion_op;
//     std::optional<FermionOperator> ref_op;

//     SECTION("Empty string") {
//         fermion_op = FermionOperator::loads("");
//     }
//     SECTION("Only whitespace") {
//         fermion_op = FermionOperator::loads("      ");
//     }
//     SECTION(R"s(Invalid: ('{"": ""}'))s") {
//         fermion_op = FermionOperator::loads(R"s({"": ""})s");
//     }
//     SECTION(R"s(Invalid: ('{"X1 Y2": "1"}'))s") {
//         fermion_op = FermionOperator::loads(R"s({"X1 Y2": "1"})s");
//     }
//     SECTION(R"s(Invalid: ('"1" : "(1+2.1j)"'))s") {
//         fermion_op = FermionOperator::loads(R"s("1" : "(1+2.1j)")s");
//     }

//     SECTION(R"s({"": "1.23"})s") {
//         fermion_op = FermionOperator::loads(R"({"": "1.23"})");
//         ref_op = FermionOperator::identity() * 1.23;
//     }
//     SECTION(R"s({"": "2.34j"})s") {
//         fermion_op = FermionOperator::loads(R"({"": "2.34j"})");
//         ref_op = FermionOperator::identity() * 2.34i;
//     }
//     SECTION(R"s({"": "(3-2j)"})s") {
//         fermion_op = FermionOperator::loads(R"s({"": "(3-2j)"})s");
//         ref_op = FermionOperator::identity() * (3. - 2.i);
//     }
//     SECTION(R"s({"1": "(1+2.1j)"})s") {
//         fermion_op = FermionOperator::loads(R"s({"1": "(1+2.1j)"})s");
//         ref_op = FermionOperator("1", 1.0 + 2.1i);
//     }
//     SECTION(R"s({"2 1^ 3 4^": "(1+2j)"})s") {
//         fermion_op = FermionOperator::loads(R"s({"2 1^ 3 4^": "(1+2j)"})s");
//         ref_op = FermionOperator("2 1^ 3 4^", 1.0 + 2.i);
//     }
//     SECTION(R"s({"2 3^ 4^": "4.5j", "1" : "1"})s") {
//         fermion_op = FermionOperator::loads(
//             R"s({"2 3^ 4^": "4.5j",
//                  "1" : "1"})s");
//         ref_op = FermionOperator("2 3^ 4^", 4.5i) + FermionOperator("1");
//     }

//     if (ref_op) {
//         REQUIRE(fermion_op.has_value());
//         CHECK(fermion_op.value() == ref_op.value());
//     } else {
//         REQUIRE(!fermion_op.has_value());
//     }
// }

// TEST_CASE("QubitOperator loads", "[terms_op][ops]") {
//     std::string json_data;
//     std::optional<QubitOperator> qubit_op;
//     std::optional<QubitOperator> ref_op;

//     SECTION("Empty string") {
//         qubit_op = QubitOperator::loads("");
//     }
//     SECTION("Only whitespace") {
//         qubit_op = QubitOperator::loads("      ");
//     }
//     SECTION(R"s(Invalid: ('{"": ""}'))s") {
//         qubit_op = QubitOperator::loads(R"s({"": ""})s");
//     }
//     SECTION(R"s(Invalid: ('{"1 2^": "1"}'))s") {
//         qubit_op = QubitOperator::loads(R"s({"1 2^": "1"})s");
//     }
//     SECTION(R"s(Invalid: ('"X1" : "(1+2.1j)"'))s") {
//         qubit_op = QubitOperator::loads(R"s("X1" : "(1+2.1j)")s");
//     }

//     SECTION(R"s({"": "1.23"})s") {
//         qubit_op = QubitOperator::loads(R"({"": "1.23"})");
//         ref_op = QubitOperator::identity() * 1.23;
//     }
//     SECTION(R"s({"": "2.34j"})s") {
//         qubit_op = QubitOperator::loads(R"({"": "2.34j"})");
//         ref_op = QubitOperator::identity() * 2.34i;
//     }
//     SECTION(R"s({"": "(3-2j)"})s") {
//         qubit_op = QubitOperator::loads(R"s({"": "(3-2j)"})s");
//         ref_op = QubitOperator::identity() * (3. - 2.i);
//     }
//     SECTION(R"s({"X1": "(1+2.1j)"})s") {
//         qubit_op = QubitOperator::loads(R"s({"X1": "(1+2.1j)"})s");
//         ref_op = QubitOperator("X1", 1.0 + 2.1i);
//     }
//     SECTION(R"s({"X2 Y1 Z3 Y4": "(1+2j)"})s") {
//         qubit_op = QubitOperator::loads(R"s({"X2 Y1 Z3 Y4": "(1+2j)"})s");
//         ref_op = QubitOperator("X2 Y1 Z3 Y4", 1.0 + 2.i);
//     }
//     SECTION(R"s({"X2 Z3 Y4": "4.5j", "X1" : "1"})s") {
//         qubit_op = QubitOperator::loads(
//             R"s({"X2 Z3 Y4": "4.5j",
//                  "X1" : "1"})s");
//         ref_op = QubitOperator("X2 Z3 Y4", 4.5i) + QubitOperator("X1");
//     }

//     if (ref_op) {
//         REQUIRE(qubit_op.has_value());
//         CHECK(qubit_op.value() == ref_op.value());
//     } else {
//         REQUIRE(!qubit_op.has_value());
//     }
// }

TEST_CASE("TermsOperator JSON save - load", "[terms_op][ops]") {
    DummyOperatorCD op;
    SECTION("Identity") {
        op = DummyOperatorCD::identity() * (1.2 + 5.4i);
    }
    SECTION("1v 2^ X3 Y4 Z5 + Identity") {
        op = DummyOperatorCD(
                 terms_t{
                     {1, TermValue::a}, {2, TermValue::adg}, {3, TermValue::X}, {4, TermValue::Y}, {5, TermValue::Z}},
                 1.2i)
             + DummyOperatorCD::identity();
    }

    CHECK(op == DummyOperatorCD::loads(op.dumps()));
}

// =============================================================================

TEST_CASE("TermsOperator arithmetic operators (+)", "[terms_op][ops]") {
    using value_t = coeff_term_dict_t::value_type;

    SECTION("Addition (TermsOperator)") {
        DummyOperatorCD op;
        coeff_term_dict_t ref_terms;
        auto& ordered_index = ref_terms.get<order::insertion>();

        CHECK(std::empty(op));

        op += DummyOperatorCD{};
        CHECK(std::empty(op));

        // op = {'X3': 2.3}
        auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
        REQUIRE(inserted1);
        op += DummyOperatorCD(it1->first, it1->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3,  'X1': 1.}
        auto [it2, inserted2] = ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 1.);
        REQUIRE(inserted2);
        op += DummyOperatorCD(it2->first, it2->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 + 1.85i,  'X1': 1.}
        const auto term3 = it1->first.front();
        const auto coeff3 = 1.85i;
        ordered_index.modify(it1, [&coeff3](value_t& value) { value.second += coeff3; });

        op += DummyOperatorCD(term3, coeff3);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 + 1.85i,  'X1': 1.,  'Y1': 10.}
        auto [it3, inserted3] = ref_terms.emplace_back(terms_t{{1, TermValue::Y}}, 10.);
        REQUIRE(inserted3);
        op += DummyOperatorCD(it3->first, it3->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 3);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 + 1.85i,  'Y1': 10.}
        const auto term4 = it2->first.front();
        const auto coeff4 = -it2->second;
        ref_terms.erase(it2);
        op += DummyOperatorCD(term4, coeff4);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);
    }

    SECTION("Addition (numbers)") {
        coeff_term_dict_t ref_terms;
        auto& ordered_index = ref_terms.get<order::insertion>();

        auto [it, inserted] = ref_terms.emplace_back(terms_t{}, 1.);
        DummyOperatorCD op = DummyOperatorCD::identity();

        CHECK(!std::empty(op));
        REQUIRE(op.is_identity());
        CHECK(op.get_terms() == ref_terms);

        // SECTION("Integer") {
        //     UNSCOPED_INFO("Integer");
        //     const auto addend = 13;
        //     op += addend;
        //     it->second += addend;
        // }

        SECTION("Double") {
            UNSCOPED_INFO("Double");
            const auto addend = 2.34;
            op += addend;
            ordered_index.modify(it, [&addend](value_t& value) { value.second += addend; });
        }

        SECTION("Complex double") {
            UNSCOPED_INFO("Complex double");
            const auto addend = 5. + 2.34i;
            op += addend;
            ordered_index.modify(it, [&addend](value_t& value) { value.second += addend; });
        }

        // NB: important to have this check here first (UNSCOPED_INFO)
        CHECK(op.get_terms() == ref_terms);
        CHECK(!std::empty(op));
        REQUIRE(op.is_identity());
    }
}

TEST_CASE("TermsOperator arithmetic operators (-)", "[terms_op][ops]") {
    using value_t = coeff_term_dict_t::value_type;

    SECTION("Subtraction (TermsOperator)") {
        // NB: careful with the signs of coefficients!
        DummyOperatorCD op;
        coeff_term_dict_t ref_terms;
        auto& ordered_index = ref_terms.get<order::insertion>();

        CHECK(std::empty(op));

        op -= DummyOperatorCD{};
        CHECK(std::empty(op));

        // op = {'X3': 2.3}
        auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
        REQUIRE(inserted1);
        op -= DummyOperatorCD(it1->first, -it1->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3,  'X1': 1.}
        auto [it2, inserted2] = ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 1.);
        REQUIRE(inserted2);
        op -= DummyOperatorCD(it2->first, -it2->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 - 1.85i,  'X1': 1.}
        const auto term3 = it1->first.front();
        const auto coeff3 = 1.85i;
        ordered_index.modify(it1, [&coeff3](value_t& value) { value.second -= coeff3; });
        op -= DummyOperatorCD(term3, coeff3);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 - 1.85i,  'X1': 1.,  'Y1': 10.}
        auto [it3, inserted3] = ref_terms.emplace_back(terms_t{{1, TermValue::Y}}, 10.);
        REQUIRE(inserted3);
        op -= DummyOperatorCD(it3->first, -it3->second);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 3);
        CHECK(op.get_terms() == ref_terms);

        // op = {'X3': 2.3 - 1.85i,  'Y1': 10.}
        const auto term4 = it2->first.front();
        const auto coeff4 = it2->second;
        ref_terms.erase(it2);
        op -= DummyOperatorCD(term4, coeff4);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);
    }

    SECTION("Subtraction (numbers)") {
        coeff_term_dict_t ref_terms;
        auto& ordered_index = ref_terms.get<order::insertion>();

        auto [it, inserted] = ref_terms.emplace_back(terms_t{}, 1.);
        DummyOperatorCD op = DummyOperatorCD::identity();

        CHECK(!std::empty(op));
        REQUIRE(op.is_identity());
        CHECK(op.get_terms() == ref_terms);

        // SECTION("Integer") {
        //     const auto subtrahend = 13;
        //     SECTION("In-place subtraction") {
        //         UNSCOPED_INFO("In-place subtraction");
        //         op -= subtrahend;
        //     }
        //     SECTION("Left subtraction") {
        //         UNSCOPED_INFO("Left subtraction");
        //         op = op - subtrahend;
        //     }
        //     SECTION("Right subtraction") {
        //         UNSCOPED_INFO("Right subtraction");
        //         op = subtrahend - op;
        //     }

        //     it->second -= subtrahend;
        // }

        SECTION("Double") {
            const auto subtrahend = 2.34;
            SECTION("In-place subtraction") {
                UNSCOPED_INFO("In-place subtraction");
                op -= subtrahend;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second -= subtrahend; });
            }
            SECTION("Left subtraction") {
                UNSCOPED_INFO("Left subtraction");
                op = op - subtrahend;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second -= subtrahend; });
            }
            SECTION("Right subtraction") {
                UNSCOPED_INFO("Right subtraction");
                op = subtrahend - op;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second = subtrahend - value.second; });
            }
        }

        SECTION("Complex double") {
            const auto subtrahend = 5. + 2.34i;
            SECTION("In-place subtraction") {
                UNSCOPED_INFO("In-place subtraction");
                op -= subtrahend;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second -= subtrahend; });
            }
            SECTION("Left subtraction") {
                UNSCOPED_INFO("Left subtraction");
                op = op - subtrahend;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second -= subtrahend; });
            }
            SECTION("Right subtraction") {
                UNSCOPED_INFO("Right subtraction");
                op = subtrahend - op;
                ordered_index.modify(it, [&subtrahend](value_t& value) { value.second = subtrahend - value.second; });
            }
        }

        // NB: important to have this check here first (UNSCOPED_INFO)
        CHECK(op.get_terms() == ref_terms);
        CHECK(!std::empty(op));
        REQUIRE(op.is_identity());
    }
}

TEST_CASE("TermsOperator arithmetic operators (*)", "[terms_op][ops]") {
    using value_t = coeff_term_dict_t::value_type;

    SECTION("Multiplication (TermsOperator)") {
        DummyOperatorCD op;
        coeff_term_dict_t ref_terms;

        // op = {'X3': 2.3}
        auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
        REQUIRE(inserted1);
        op += DummyOperatorCD(it1->first, it1->second);

        // op = {'X3 Y1': 4.255}
        const auto term = term_t{1, TermValue::Y};
        const auto coeff = 1.85;
        auto [it2, inserted2] = ref_terms.emplace_back(terms_t{it1->first.front(), term}, it1->second * coeff);
        REQUIRE(inserted2);
        ref_terms.erase(it1);

        op *= DummyOperatorCD({term}, coeff);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 1);
        CHECK(op.get_terms() == ref_terms);

        /* NB: cannot test for the case where the product of terms is already present in the original since we do not
         *     implement the simplify_ member function in these tests.
         */
    }

    SECTION("Multiplication (numbers)") {
        coeff_term_dict_t ref_terms{
            {terms_t{{0, TermValue::X}}, 2.34},
            {terms_t{{3, TermValue::Z}}, 2.3i},
        };
        DummyOperatorCD op{ref_terms};

        CHECK(!std::empty(op));
        CHECK(op.get_terms() == ref_terms);

        const auto do_multiply = [](auto& terms, const auto& multiplier) constexpr {
            auto& ordered_index = terms.template get<order::insertion>();
            for (auto it(begin(terms)), it_end(end(terms)); it != it_end; ++it) {
                ordered_index.modify(it, [&multiplier](value_t& value) { value.second *= multiplier; });
            }
            // for (auto& [_, coeff] : terms) {
            //     coeff *= multiplier;
            // }
        };

        // SECTION("Integer") {
        //     UNSCOPED_INFO("Integer");
        //     const auto multiplier = 13;
        //     op *= multiplier;
        //     do_multiply(ref_terms, multiplier);
        // }

        SECTION("Double") {
            UNSCOPED_INFO("Double");
            const auto multiplier = 2.34;
            op *= multiplier;
            do_multiply(ref_terms, multiplier);
        }

        SECTION("Complex double") {
            UNSCOPED_INFO("Complex double");
            const auto multiplier = 5. + 2.34i;
            op *= multiplier;
            do_multiply(ref_terms, multiplier);
        }

        // NB: important to have this check here first (UNSCOPED_INFO)
        CHECK(op.get_terms() == ref_terms);
        CHECK(!std::empty(op));
    }
}

TEST_CASE("TermsOperator arithmetic operators (/)", "[terms_op][ops]") {
    using value_t = coeff_term_dict_t::value_type;

    SECTION("Division (numbers)") {
        coeff_term_dict_t ref_terms{
            {terms_t{{0, TermValue::X}}, 2.34},
            {terms_t{{3, TermValue::Z}}, 2.3i},
        };

        DummyOperatorCD op{ref_terms};

        CHECK(!std::empty(op));
        CHECK(op.get_terms() == ref_terms);

        const auto do_multiply = [](auto& terms, const auto& multiplier) constexpr {
            auto& ordered_index = terms.template get<order::insertion>();
            for (auto it(begin(ordered_index)), it_end(end(ordered_index)); it != it_end; ++it) {
                ordered_index.modify(it, [&multiplier](value_t& value) { value.second /= multiplier; });
            }
            // for (auto& [_, coeff] : terms) {
            //     coeff /= multiplier;
            // }
        };

        const auto approx_equal = [&ref_terms](const coeff_term_dict_t& other_terms) {
            for (const auto& [ref, other] : boost::combine(ref_terms, other_terms)) {
                const auto& [local_ops_ref, coeff_ref] = ref;
                const auto& [local_ops_other, coeff_other] = other;
                if (local_ops_ref != local_ops_other) {
                    FAIL_CHECK(fmt::format("{} != {}", local_ops_ref, local_ops_other));
                    return false;
                }
                const auto rel_diff = std::abs((coeff_ref - coeff_other)
                                               / std::min(std::abs(coeff_ref), std::abs(coeff_other)));
                if (rel_diff > 1.e-8) {
                    FAIL_CHECK(fmt::format("{} != {} ({})", coeff_ref, coeff_other, rel_diff));
                    return false;
                }
            }
            return true;
        };

        // SECTION("Integer") {
        //     UNSCOPED_INFO("Integer");
        //     const auto divisor = 13;
        //     op /= divisor;
        //     do_multiply(ref_terms, divisor);
        // }

        SECTION("Double") {
            UNSCOPED_INFO("Double");
            const auto divisor = 2.34;
            op /= divisor;
            do_multiply(ref_terms, divisor);
        }

        SECTION("Complex double") {
            UNSCOPED_INFO("Complex double");
            const auto divisor = 5. + 2.34i;
            op /= divisor;
            do_multiply(ref_terms, divisor);
        }

        CHECK_THAT(op.get_terms(), Catch::Matchers::Predicate<coeff_term_dict_t>(approx_equal, "Approx equal terms"));
        CHECK(!std::empty(op));
    }
}

TEST_CASE("TermsOperator unary operators", "[terms_op][ops]") {
    using value_t = coeff_term_dict_t::value_type;

    SECTION("Unary minus (TermsOperator)") {
        coeff_term_dict_t ref_terms;

        auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
        auto [it2, inserted2] = ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 1.);
        REQUIRE(inserted1);
        REQUIRE(inserted2);

        // op = {'X3': 2.3,  'X1': 1.}
        const DummyOperatorCD op(ref_terms);
        const DummyOperatorCD ref = op;
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        const auto result = -op;
        auto& ordered_index = ref_terms.get<order::insertion>();
        for (auto it(begin(ordered_index)), it_end(end(ordered_index)); it != it_end; ++it) {
            ordered_index.modify(it, [](value_t& value) { value.second *= -1; });
        }

        REQUIRE(!std::empty(result));
        REQUIRE(op == ref);
        REQUIRE(op != result);
        CHECK(std::size(result) == std::size(ref));
        CHECK(result.get_terms() == ref_terms);
    }
}

TEST_CASE("TermsOperator mathmetic operators (pow)", "[terms_op][ops]") {
    SECTION("Addition (TermsOperator)") {
        coeff_term_dict_t ref_terms;

        auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
        auto [it2, inserted2] = ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 1.);
        REQUIRE(inserted1);
        REQUIRE(inserted2);

        // op = {'X3': 2.3,  'X1': 1.}
        DummyOperatorCD op(ref_terms);
        REQUIRE(!std::empty(op));
        CHECK(std::size(op) == 2);
        CHECK(op.get_terms() == ref_terms);

        const auto result = op.pow(3);
        const auto ref = op * op * op;

        REQUIRE(!std::empty(result));
        CHECK(std::size(result) == std::size(ref));
        CHECK(result.get_terms() == ref.get_terms());
    }
}

TEST_CASE("TermsOperator split", "[terms_op][ops]") {
    const auto lhs = DummyOperatorCD(terms_t{{3, TermValue::X}}, 1.2i);
    const auto rhs = DummyOperatorCD(terms_t{{1, TermValue::Z}}, 1.2);
    const auto qubit_op = lhs + rhs;

    const auto get = [](const auto& op) { return op.first * op.second; };

    const auto splitted_operator = qubit_op.split();
    REQUIRE(std::size(splitted_operator) == 2);
    if (get(splitted_operator[0]) == lhs) {
        CHECK(get(splitted_operator[0]) == lhs);
        CHECK(get(splitted_operator[1]) == rhs);
    } else {
        CHECK(get(splitted_operator[0]) == rhs);
        CHECK(get(splitted_operator[1]) == lhs);
    }
}

TEST_CASE("TermsOperator comparison operators", "[terms_op][ops]") {
    coeff_term_dict_t ref_terms;

    auto [it1, inserted1] = ref_terms.emplace_back(terms_t{{3, TermValue::X}}, 2.3);
    auto [it2, inserted2] = ref_terms.emplace_back(terms_t{{1, TermValue::X}}, 1.);
    REQUIRE(inserted1);
    REQUIRE(inserted2);

    // op = {'X3': 2.3,  'X1': 1.}
    const DummyOperatorCD op(ref_terms);
    DummyOperatorCD other(ref_terms);

    CHECK(op == op);
    CHECK(op == other);

    SECTION("Add identity term") {
        other += DummyOperatorCD::identity();
    }
    SECTION("Add other term") {
        other += DummyOperatorCD{terms_t{{2, TermValue::Z}}, 2.34i};
    }
    SECTION("No common terms") {
        other = DummyOperatorCD{terms_t{{2, TermValue::Z}}, 2.34i};
    }
    CHECK(!(op == other));
    CHECK(op != other);
}

// =============================================================================
