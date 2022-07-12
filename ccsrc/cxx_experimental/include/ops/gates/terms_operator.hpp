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

#ifndef TERMS_OPERATOR_HPP
#define TERMS_OPERATOR_HPP

#include <complex>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include <boost/operators.hpp>

#include "core/config.hpp"

#include "core/traits.hpp"
#include "core/types.hpp"
#include "ops/meta/dagger.hpp"
#if MQ_HAS_CONCEPTS
#    include "core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::ops {
using namespace std::literals::complex_literals;  // NOLINT(build/namespaces_literals)
enum class TermValue : uint8_t {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
    a = 0,
    adg = 1,
};

#if MQ_HAS_CONCEPTS
#    define TYPENAME_NUMBER concepts::number
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF
#    define TYPENAME_NUMBER_CONSTRAINTS_IMPL
#else
#    define TYPENAME_NUMBER typename
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF                                                                            \
        , typename = std::enable_if_t < traits::is_complex_v<number_t> || std::is_floating_point_v < number_t >>
#    define TYPENAME_NUMBER_CONSTRAINTS_IMPL , typename
#endif  // MQ_HAS_CONCEPTS

template <typename derived_t>
class TermsOperator
    // clang-format off
    : boost::additive1<TermsOperator<derived_t>
    , boost::additive2<TermsOperator<derived_t>, double
    , boost::additive2<TermsOperator<derived_t>, std::complex<double>
    , boost::multipliable1<TermsOperator<derived_t>
    , boost::multiplicative2<TermsOperator<derived_t>, double
    , boost::multiplicative2<TermsOperator<derived_t>, std::complex<double>
      >>>>>> {
    // clang-format on
 public:
    using non_const_num_targets = void;
    using self_t = TermsOperator<derived_t>;

    static constexpr auto EQ_TOLERANCE = 1.e-8;

    using coefficient_t = std::complex<double>;
    using term_t = std::pair<uint32_t, TermValue>;
    using complex_term_dict_t = std::map<std::vector<term_t>, coefficient_t>;

    static constexpr std::string_view kind() {
        return "mindquantum.terms_operator";
    }

    TermsOperator() = default;

    explicit TermsOperator(term_t term, coefficient_t coefficient = 1.0);

    explicit TermsOperator(complex_term_dict_t terms);

    //! Return the adjoint of this operator
    MQ_NODISCARD operator_t adjoint() const noexcept;

    //! Compress a TermsOperator
    /*!
     * Eliminate terms with small coefficients close to zero. Also remove small imaginary and real parts.
     *
     * Compression is done /e in-place and the modified operator is then returned.
     *
     * \param abs_tol Absolute tolerance, must be a positive non-zero number
     */
    self_t& compress(double abs_tol = EQ_TOLERANCE);

    //! Power operator (self-multiply n-times)
    self_t pow(uint32_t exponent) const;

    // =================================

    //! In-place addition of another terms operator
    self_t& operator+=(const self_t& other);

    //! In-place addition with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    self_t& operator+=(const number_t& number);

    // ---------------------------------

    //! In-place subtraction of another terms operator
    self_t& operator-=(const self_t& other);

    //! In-place subtraction with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    self_t& operator-=(const number_t& number);

    // ---------------------------------

    //! Unary - operator with a number
    self_t operator-() const;

    // ---------------------------------

    //! In-place multiplication of another terms operator
    self_t& operator*=(const self_t& other);

    //! In-place multiplication with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    self_t& operator*=(const number_t& number);

    // ---------------------------------

    //! In-place multiplication with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    self_t& operator/=(const number_t& number);

    // =================================

    bool operator==(self_t& other);

    bool operator==(const self_t& other) const;

 private:
    template <typename assign_modify_op_t, typename coeff_unary_op_t>
    self_t& add_sub_impl_(const self_t& other, assign_modify_op_t&& assign_modify_op,
                          coeff_unary_op_t&& coeff_unary_op);

    complex_term_dict_t terms_;
};

}  // namespace mindquantum::ops

#include "ops/gates/terms_operator.tpp"

#endif /* TERMS_OPERATOR_HPP */
