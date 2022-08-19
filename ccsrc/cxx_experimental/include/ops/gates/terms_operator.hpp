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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/operators.hpp>

#include "core/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS
#include <nlohmann/json.hpp>

#include "core/traits.hpp"
#include "core/types.hpp"
#include "ops/gates/details/complex_double_coeff_policy.hpp"
#include "ops/meta/dagger.hpp"

namespace mindquantum::traits {
template <typename T, typename = void>
struct is_terms_operator : std::false_type {};

template <typename T>
struct is_terms_operator<T, std::void_t<typename T::terms_operator_tag>> : std::true_type {};
}  // namespace mindquantum::traits

namespace mindquantum::ops {
enum class TermValue : uint8_t {
    I = 10,
    X = 11,
    Y = 12,
    Z = 13,
    a = 0,
    adg = 1,
};

// NOLINTNEXTLINE(*avoid-c-arrays,readability-identifier-length)
NLOHMANN_JSON_SERIALIZE_ENUM(TermValue, {
                                            {TermValue::I, "I"},
                                            {TermValue::X, "X"},
                                            {TermValue::Y, "Y"},
                                            {TermValue::Z, "Z"},
                                            {TermValue::a, "v"},
                                            {TermValue::adg, "^"},
                                        });

using term_t = std::pair<uint32_t, TermValue>;
using terms_t = std::vector<term_t>;

using py_term_t = std::pair<uint32_t, uint32_t>;
using py_terms_t = std::vector<py_term_t>;

template <typename coefficient_t>
using term_dict_t = std::map<std::vector<term_t>, coefficient_t>;

#if MQ_HAS_CONCEPTS
#    define TYPENAME_NUMBER concepts::number
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF_ADD(x)
#    define TYPENAME_NUMBER_CONSTRAINTS_IMPL
#else
#    define TYPENAME_NUMBER typename
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF                                                                            \
        , typename = std::enable_if_t < traits::is_complex_v<number_t> || std::is_floating_point_v < number_t >>
#    define TYPENAME_NUMBER_CONSTRAINTS_DEF_ADD(x)                                                                     \
        , typename = std::enable_if_t<(traits::is_complex_v<number_t> || std::is_floating_point_v<number_t>) &&(x)>
#    define TYPENAME_NUMBER_CONSTRAINTS_IMPL , typename
#endif  // MQ_HAS_CONCEPTS

//! Base class for term operators (like qubit or fermion operators)
/*!
 * \note This template CRTP class expects the derived classes to implement the following member functions:
 *         - static std::pair<terms_t, coefficient_t> sort_terms_(terms_t local_ops, coefficient_t coeff);
 *         - static std::tuple<std::vector<term_t>, coefficient_t> simplify_(terms_t terms, coefficient_t coeff = 1.);
 *         - static std::tuple<std::vector<term_t>, coefficient_t> simplify_(py_terms_t terms, coefficient_t coeff
 * = 1.);
 */
template <typename derived_t, typename term_policy_t_, typename coeff_policy_t_ = details::CmplxDoubleCoeffPolicy>
class TermsOperator
    // clang-format off
    : boost::additive1<derived_t
    , boost::additive2<derived_t, double
    , boost::additive2<derived_t, std::complex<double>
    , boost::multipliable1<derived_t
    , boost::multiplicative2<derived_t, double
    , boost::multiplicative2<derived_t, std::complex<double>
#if !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
    , boost::equality_comparable1<derived_t>
#endif  // !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
    >>>>>> {
    // clang-format on
 public:
    using non_const_num_targets = void;
    using terms_operator_tag = void;
    using term_policy_t = term_policy_t_;
    using coeff_policy_t = coeff_policy_t_;
    using base_t = TermsOperator<derived_t, term_policy_t, coeff_policy_t>;

    static constexpr auto EQ_TOLERANCE = coeff_policy_t::EQ_TOLERANCE;

    using coefficient_t = typename coeff_policy_t::coeff_t;
    using term_t = mindquantum::ops::term_t;
    using terms_t = mindquantum::ops::terms_t;
    using coeff_term_dict_t = term_dict_t<coefficient_t>;
    using py_coeff_term_list_t = std::vector<std::pair<mindquantum::ops::py_terms_t, coefficient_t>>;

    static constexpr std::string_view kind() {
        return "mindquantum.terms_operator";
    }

    TermsOperator() = default;
    TermsOperator(const TermsOperator&) = default;
    TermsOperator(TermsOperator&&) noexcept = default;
    TermsOperator& operator=(const TermsOperator&) = default;
    TermsOperator& operator=(TermsOperator&&) noexcept = default;
    ~TermsOperator() noexcept = default;

    explicit TermsOperator(term_t term, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperator(const terms_t& terms, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperator(const py_terms_t& terms, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperator(const coeff_term_dict_t& terms);

    //! Constructor from a string representing a list of terms
    /*!
     * \note If parsing the string fails, the resulting TermsOperator object will represent the identity. If logging is
     *       enabled, an error message will be printed inside the log with an appropriate error message.
     */
    explicit TermsOperator(std::string_view terms_string, coefficient_t coeff = 1.0);

    //! Return the number of target qubits of an operator
    MQ_NODISCARD uint32_t num_targets() const;

    //! Return true if the TermsOperator is empty
    MQ_NODISCARD auto empty() const noexcept;

    //! Return the number of terms within the TermsOperator
    MQ_NODISCARD auto size() const;

    MQ_NODISCARD const coeff_term_dict_t& get_terms() const;

    //! Return the operator in list of term and coefficient pair.
    // TODO(xusheng): Since the key of coeff_term_dict_t is list, which
    // is unhashable in python, so the get_terms is unable to bind.
    MQ_NODISCARD py_coeff_term_list_t get_terms_pair() const;

    //! Check whether this operator is equivalent to the identity
    MQ_NODISCARD bool is_identity(double abs_tol = EQ_TOLERANCE) const;

    //! Calculate the number of qubits on which an operator acts
    MQ_NODISCARD term_t::first_type count_qubits() const;

    MQ_NODISCARD static derived_t identity();

    // -------------------------------------------------------------------------

    //! Return the hermitian of this operator
    MQ_NODISCARD derived_t hermitian() const;

    //! Return the adjoint of this operator
    MQ_NODISCARD operator_t adjoint() const noexcept;

    //! Get the coefficient of the constant (ie. identity) term
    MQ_NODISCARD coefficient_t constant() const noexcept;

    //! Set the coefficient of the constant (ie. identity) term
    void constant(const coefficient_t& coeff);

    //! Test whether this operator consists of a single term
    MQ_NODISCARD bool is_singlet() const;

    //! Split a single-term operator into its components/words
    /*!
     * \note The coefficient attribute is ignored.
     *
     * \return A vector of TermsOperator::derived_t
     */
    MQ_NODISCARD std::vector<derived_t> singlet() const;

    //! Get the coefficient of a single-term operator
    MQ_NODISCARD coefficient_t singlet_coeff() const;

    //! Split the operator into its individual components
    MQ_NODISCARD std::vector<std::pair<coefficient_t, derived_t>> split() const;

    //! Return a copy of an operator with all the coefficients set to their real part
    MQ_NODISCARD derived_t real() const;

    //! Return a copy of an operator with all the coefficients set to their imaginary part
    MQ_NODISCARD derived_t imag() const;

    //! Compress a TermsOperator
    /*!
     * Eliminate terms with small coefficients close to zero. Also remove small imaginary and real parts.
     *
     * Compression is done \e in-place and the modified operator is then returned.
     *
     * \param abs_tol Absolute tolerance, must be a positive non-zero number
     * \return The compressed term operator
     *
     * \note Example
     * \code{.cpp}
     *    ham_compress = QubitOperator("X0 Y3", 0.5) + QubitOperator("X0 Y2", 1e-7);
     *    // ham_compress = 1/10000000 [X0 Y2] + 1/2 [X0 Y3]
     *    ham_compress.compress(1e-6);
     *    // ham_compress = 1/2 [X0 Y3]
     * \endcode
     */
    derived_t& compress(double abs_tol = EQ_TOLERANCE);

    // =========================================================================

    //! Convert a TermsOperator to a string
    MQ_NODISCARD std::string to_string() const noexcept;

    //! Dump TermsOperator into JSON(JavaScript Object Notation).
    /*!
     * \param indent Number of spaces to use for indent
     * \return JSON formatted string
     */
    MQ_NODISCARD std::string dumps(std::size_t indent = 4UL) const;

    //! Load a TermsOperator from a JSON-formatted string.
    /*!
     * \param string_data
     * \return A TermsOperator if the loading was successful, false otherwise.
     */
    MQ_NODISCARD static std::optional<derived_t> loads(std::string_view string_data);

    // =========================================================================

    //! In-place addition of another terms operator
    derived_t& operator+=(const derived_t& other);

    //! In-place addition with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    derived_t& operator+=(const number_t& number);

    // ---------------------------------

    //! In-place subtraction of another terms operator
    derived_t& operator-=(const derived_t& other);

    //! In-place subtraction with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    derived_t& operator-=(const number_t& number);

    // ---------------------------------

    //! Unary - operator with a number
    MQ_NODISCARD derived_t operator-() const;

    // ---------------------------------

    //! In-place multiplication of another terms operator
    derived_t& operator*=(const derived_t& other);

    //! In-place multiplication with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    derived_t& operator*=(const number_t& number);

    //! In-place multiplication with a coefficient_t
    derived_t& operator*=(const coefficient_t& coeff_out);

    // ---------------------------------

    //! In-place multiplication with a number
    template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_DEF>
    derived_t& operator/=(const number_t& number);

    // ---------------------------------

    //! Power operator (self-multiply n-times)
    MQ_NODISCARD derived_t pow(uint32_t exponent) const;

    // =================================

    // TODO(dnguyen): This might be counter-intuitive...
    //! Comparison operator (non-const)
    /*!
     * Compared to its const counterpart, this overload calls \c compress() before performing th<e comparison.
     *
     * \param other Another term operators
     */
    // MQ_NODISCARD bool operator==(derived_t& other);

    //! Comparison operator
    /*!
     * \param other Another term operators
     */
    MQ_NODISCARD bool operator==(const derived_t& other) const;

 protected:
    struct sorted_constructor_t {
    } sorted_constructor;

    //! (internal) Constructor designed for cases where the terms dictionary is already optimal
    TermsOperator(coeff_term_dict_t terms, sorted_constructor_t /* unused */);

 private:
    //! Addition/subtraction helper member function
    template <typename assign_modify_op_t, typename coeff_unary_op_t>
    derived_t& add_sub_impl_(const derived_t& other, assign_modify_op_t&& assign_modify_op,
                             coeff_unary_op_t&& coeff_unary_op);

    //! Calculate the number of target qubits of a TermOperator.
    void calculate_num_targets_() noexcept;

 protected:
    uint32_t num_targets_{0UL};
    coeff_term_dict_t terms_;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(base_t, num_targets_, terms_);
};

template <TYPENAME_NUMBER number_t,
          typename derived_t TYPENAME_NUMBER_CONSTRAINTS_DEF_ADD(traits::is_terms_operator<derived_t>::value)>
MQ_NODISCARD auto operator-(const number_t& number, const derived_t& other);

}  // namespace mindquantum::ops

#include "ops/gates/terms_operator.tpp"

#endif /* TERMS_OPERATOR_HPP */
