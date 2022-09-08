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

#ifndef TERMS_OPERATOR_BASE_HPP
#define TERMS_OPERATOR_BASE_HPP

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/container_hash/extensions.hpp>

#include <nlohmann/json.hpp>

#include "config/real_cast.hpp"
#include "config/tsl_ordered_map.hpp"
#include "config/type_traits.hpp"

#include "experimental/core/config.hpp"
#include "experimental/core/traits.hpp"
#include "experimental/core/types.hpp"
#include "experimental/ops/gates/details/coeff_policy.hpp"
#include "experimental/ops/gates/traits.hpp"
#include "experimental/ops/meta/dagger.hpp"

#if MQ_HAS_CONCEPTS
#    include "config/concepts.hpp"

#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

namespace mindquantum::traits {
template <typename scalar_t, bool is_real, typename = void>
struct is_compatible_scalar : std::false_type {};

// NB: If the ref coefficient is complex, then we accept all scalar types
template <typename scalar_t>
struct is_compatible_scalar<scalar_t, false, std::enable_if_t<is_termsop_number_v<std::remove_cvref_t<scalar_t>>>>
    : std::true_type {};

// NB: If the ref coefficient is real-valued, then we accept only real-valued scalar types
template <typename scalar_t>
struct is_compatible_scalar<scalar_t, true, std::enable_if_t<std::is_floating_point_v<std::remove_cvref_t<scalar_t>>>>
    : std::true_type {};

template <typename scalar_t, bool is_real>
inline constexpr auto is_compatible_scalar_v = is_compatible_scalar<scalar_t, is_real>::value;

// Real numbers
static_assert(is_compatible_scalar_v<float, true>);
static_assert(is_compatible_scalar_v<double, true>);
static_assert(!is_compatible_scalar_v<std::complex<float>, true>);
static_assert(!is_compatible_scalar_v<std::complex<double>, true>);
// Complex numbers
static_assert(is_compatible_scalar_v<float, false>);
static_assert(is_compatible_scalar_v<double, false>);
static_assert(is_compatible_scalar_v<std::complex<float>, false>);
static_assert(is_compatible_scalar_v<std::complex<double>, false>);

// -------------------------------------

template <typename T, typename = void>
struct is_terms_operator : std::false_type {};

template <typename T>
struct is_terms_operator<T, std::void_t<typename std::remove_cvref_t<T>::terms_operator_tag>> : std::true_type {};

template <typename T>
inline constexpr auto is_terms_operator_v = is_terms_operator<T>::value;
}  // namespace mindquantum::traits

// -----------------------------------------------------------------------------

#if MQ_HAS_CONCEPTS
namespace mindquantum::concepts {
template <typename op_t>
concept terms_op = requires(op_t) {
    typename std::remove_cvref_t<op_t>::terms_operator_tag;
    { std::remove_cvref_t<op_t>::is_real_valued } -> same_decay_as<bool>;
};

template <typename op_t>
concept not_terms_op = !terms_op<op_t>;

template <typename T, typename U>
concept is_not_same_as = !std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<U>>;

template <typename op_t, typename ref_op_t>
concept compat_terms_op = requires(op_t, ref_op_t) {
    requires terms_op<op_t>;
    requires terms_op<ref_op_t>;
    requires traits::is_compatible_scalar_v<typename std::remove_cvref_t<op_t>::coefficient_t,
                                            std::remove_cvref_t<ref_op_t>::is_real_valued>;
};

template <typename scalar_t, typename ref_op_t>
concept compat_terms_op_scalar = requires(scalar_t, ref_op_t) {
    requires traits::is_compatible_scalar_v<std::remove_cvref_t<scalar_t>,
                                            std::remove_cvref_t<ref_op_t>::is_real_valued>;
};
}  // namespace mindquantum::concepts
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

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

// =============================================================================

using term_t = std::pair<uint32_t, TermValue>;
using terms_t = std::vector<term_t>;

using py_term_t = std::pair<uint32_t, uint32_t>;
using py_terms_t = std::vector<py_term_t>;

// NB: using boost hash since std::hash is non-copyable (at least for recent GCC and Clang versions)
template <typename coefficient_t>
using term_dict_t = tsl::ordered_map<terms_t, coefficient_t, boost::hash<terms_t>>;

// =============================================================================

//! Base class for term operators (like qubit or fermion operators)
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
class TermsOperatorBase {
    template <template <typename coeff_t> class other_derived_t, typename other_coefficient_t,
              template <typename coeff_t> class other_term_policy_t>
    friend class TermsOperatorBase;

 public:
    using non_const_num_targets = void;
    using terms_operator_tag = void;
    using coefficient_t = coefficient_t_;

    using coeff_policy_t = details::CoeffPolicy<coefficient_t>;
    using term_policy_t = term_policy_t_<coefficient_t>;
    using coefficient_real_t = typename coeff_policy_t::coeff_policy_real_t::coeff_t;

    //! "Meta-function" to generate a a new related derived terms operator with a different coefficient type
    template <typename new_coeff_t>
    using new_derived_t = derived_t_<new_coeff_t>;

    using derived_t = derived_t_<coefficient_t>;
    using derived_real_t = derived_t_<coefficient_real_t>;

    using base_t = TermsOperatorBase<derived_t_, coefficient_t, term_policy_t_>;

    static constexpr auto is_real_valued = std::is_same_v<coefficient_t, coefficient_real_t>;
    static constexpr auto EQ_TOLERANCE = coeff_policy_t::EQ_TOLERANCE;

    using term_t = mindquantum::ops::term_t;
    using terms_t = mindquantum::ops::terms_t;
    using coeff_term_dict_t = term_dict_t<coefficient_t>;
    using py_coeff_term_list_t = std::vector<std::pair<mindquantum::ops::terms_t, coefficient_t>>;

#if !MQ_SUPPORTS_EXT_DEPENDENT_CONCEPTS
 private:
    template <typename op_t, typename = void>
    struct is_compat_terms_op_ : std::false_type {};

    template <typename op_t>
    // clang-format off
    struct is_compat_terms_op_<
        op_t,
        std::enable_if_t<mindquantum::traits::is_terms_operator_v<op_t>
                         && mindquantum::traits::is_compatible_scalar_v<
                           typename std::remove_cvref_t<op_t>::coefficient_t, is_real_valued>>>
        : std::true_type {};
    // clang-format on

    template <typename op_t>
    inline static constexpr auto is_compat_terms_op_v_ = is_compat_terms_op_<op_t>::value;

 public:
#endif  // !MQ_SUPPORTS_EXT_DEPENDENT_CONCEPTS

    static constexpr std::string_view kind() {
        return "mindquantum.terms_operator";
    }

    TermsOperatorBase() = default;
    TermsOperatorBase(const TermsOperatorBase&) = default;
    TermsOperatorBase(TermsOperatorBase&&) = default;
    TermsOperatorBase& operator=(const TermsOperatorBase&) = default;
    TermsOperatorBase& operator=(TermsOperatorBase&&) = default;
    ~TermsOperatorBase() noexcept = default;

    explicit TermsOperatorBase(term_t term, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperatorBase(const terms_t& terms, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperatorBase(const py_terms_t& terms, coefficient_t coeff = coeff_policy_t::one);

    explicit TermsOperatorBase(const coeff_term_dict_t& terms);

    template <typename other_coeff_t>
    explicit TermsOperatorBase(const derived_t_<other_coeff_t>& other);

    //! Constructor from a string representing a list of terms
    /*!
     * \note If parsing the string fails, the resulting TermsOperatorBase object will represent the identity. If
     * logging is enabled, an error message will be printed inside the log with an appropriate error message.
     */
    explicit TermsOperatorBase(std::string_view terms_string, coefficient_t coeff = coeff_policy_t::one);

    //! Return the number of target qubits of an operator
    MQ_NODISCARD uint32_t num_targets() const;

    //! Return true if the TermsOperatorBase is empty
    MQ_NODISCARD auto empty() const noexcept;

    //! Return the number of terms within the TermsOperatorBase
    MQ_NODISCARD auto size() const;

    MQ_NODISCARD const coeff_term_dict_t& get_terms() const;

    //! Return the operator in list of term and coefficient pair.
    // TODO(xusheng): Since the key of coeff_term_dict_t is list, which
    // is unhashable in python, so the get_terms is unable to bind.
    MQ_NODISCARD py_coeff_term_list_t get_terms_pair() const;

    MQ_NODISCARD coefficient_t get_coeff(const terms_t& term) const;

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
     * \return A vector of TermsOperatorBase::derived_t
     */
    MQ_NODISCARD std::vector<derived_t> singlet() const;

    //! Get the coefficient of a single-term operator
    MQ_NODISCARD coefficient_t singlet_coeff() const;

    //! Split the operator into its individual components
    MQ_NODISCARD std::vector<std::pair<coefficient_t, derived_t>> split() const;

    //! Return a copy of an operator with all the coefficients set to their real part
    MQ_NODISCARD derived_real_t real() const;

    //! Return a copy of an operator with all the coefficients set to their imaginary part
    MQ_NODISCARD derived_real_t imag() const;

    //! Compress a TermsOperatorBase
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

    //! Convert a TermsOperatorBase to a string
    MQ_NODISCARD std::string to_string() const noexcept;

    //! Dump TermsOperatorBase into JSON(JavaScript Object Notation).
    /*!
     * \param indent Number of spaces to use for indent
     * \return JSON formatted string
     */
    MQ_NODISCARD std::string dumps(std::size_t indent = 4UL) const;

    //! Load a TermsOperatorBase from a JSON-formatted string.
    /*!
     * \param string_data
     * \return A TermsOperatorBase if the loading was successful, false otherwise.
     */
    MQ_NODISCARD static std::optional<derived_t> loads(std::string_view string_data);

    // =========================================================================
    // Addition

#if MQ_HAS_CONCEPTS
    //! In-place addition of another terms operator (with/without conversion)
    template <concepts::compat_terms_op<derived_t> op_t>
    derived_t& operator+=(const op_t& other);

    //! In-place addition of a coefficient (with/without conversion)
    template <concepts::compat_terms_op_scalar<derived_t> scalar_t>
    derived_t& operator+=(const scalar_t& scalar);
#else
    //! In-place addition of another terms operator or coefficient (with/without conversion)
    template <typename type_t,
              typename = std::enable_if_t<
                  is_compat_terms_op_v_<type_t> || traits::is_compatible_scalar_v<type_t, is_real_valued>>>
    derived_t& operator+=(const type_t& op_or_scalar);
#endif  // MQ_HAS_CONCEPTS

    // -------------------------------------------------------------------------
    // Subtraction

#if MQ_HAS_CONCEPTS
    //! In-place subtraction of another terms operator (with/without conversion)
    template <concepts::compat_terms_op<derived_t> op_t>
    derived_t& operator-=(const op_t& other);

    //! In-place subtraction of a coefficient (with conversion)
    template <concepts::compat_terms_op_scalar<derived_t> scalar_t>
    derived_t& operator-=(const scalar_t& scalar);
#else
    //! In-place subtraction of another terms operator or coefficient (with/without conversion)
    template <typename type_t,
              typename = std::enable_if_t<
                  is_compat_terms_op_v_<type_t> || traits::is_compatible_scalar_v<type_t, is_real_valued>>>
    derived_t& operator-=(const type_t& op_or_scalar);
#endif  // MQ_HAS_CONCEPTS

    // ---------------------------------

    //! Unary - operator with a number
    MQ_NODISCARD derived_t operator-() const;

    // -------------------------------------------------------------------------
    // Multiplication

#if MQ_HAS_CONCEPTS
    //! In-place multiplication of another terms operator (with/without conversion)
    template <concepts::compat_terms_op<derived_t> op_t>
    derived_t& operator*=(const op_t& other);

    //! In-place multiplication of a coefficient (with conversion)
    template <concepts::compat_terms_op_scalar<derived_t> scalar_t>
    derived_t& operator*=(const scalar_t& scalar);
#else
    //! In-place multiplication of another terms operator or coefficient (with/without conversion)
    template <typename type_t,
              typename = std::enable_if_t<
                  is_compat_terms_op_v_<type_t> || traits::is_compatible_scalar_v<type_t, is_real_valued>>>
    derived_t& operator*=(const type_t& op_or_scalar);
#endif  // MQ_HAS_CONCEPTS

    // -------------------------------------------------------------------------
    // Division

    //! In-place division of a coefficient (with/without conversion)
#if MQ_SUPPORTS_EXT_DEPENDENT_CONCEPTS
    template <concepts::compat_terms_op_scalar<derived_t> scalar_t>
#else
    template <typename scalar_t, typename = std::enable_if_t<traits::is_compatible_scalar_v<scalar_t, is_real_valued>>>
#endif  // MQ_SUPPORTS_EXT_DEPENDENT_CONCEPTS
    derived_t& operator/=(const scalar_t& scalar);

    // -------------------------------------------------------------------------
    // Other mathemetical member functions

    //! Power operator (self-multiply n-times)
    MQ_NODISCARD derived_t pow(uint32_t exponent) const;

    // =================================

    //! Comparison operator
    /*!
     * \param other Another term operators
     */

#if MQ_HAS_CONCEPTS
    template <concepts::compat_terms_op<derived_t> op_t>
#else
    //! In-place subtraction of another terms operator or coefficient (with/without conversion)
    template <typename op_t, typename = std::enable_if_t<is_compat_terms_op_v_<op_t>>>
#endif  // MQ_HAS_CONCEPTS
    MQ_NODISCARD bool is_equal(const op_t& other) const;

 protected:
    struct sorted_constructor_t {
    } sorted_constructor;

    //! (internal) Constructor designed for cases where the terms dictionary is already optimal
    TermsOperatorBase(coeff_term_dict_t terms, sorted_constructor_t /* unused */);

 private:
    template <RealCastType cast_type>
    derived_real_t real_cast_() const;

    //! Addition/subtraction helper member function
    template <typename other_t, typename assign_modify_op_t, typename coeff_unary_op_t>
    derived_t& add_sub_impl_(const other_t& other, assign_modify_op_t&& assign_modify_op,
                             coeff_unary_op_t&& coeff_unary_op);

    //! Multiplication helper member function
    template <typename other_t>
    derived_t& mul_impl_(const other_t& other);

    //! Calculate the number of target qubits of a TermOperator.
    void calculate_num_targets_() noexcept;

 protected:
    uint32_t num_targets_{0UL};
    coeff_term_dict_t terms_;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(base_t, num_targets_, terms_);
};

// =============================================================================
// =============================================================================
// Implement free arithmetic operators

}  // namespace mindquantum::ops

#include "experimental/ops/gates/terms_operator_base.tpp"

#include "experimental/ops/gates/terms_operator_base_external_ops.hpp"

#endif /* TERMS_OPERATOR_BASE_HPP */
