//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#ifndef QUBITOPERATOR_OP_HPP
#define QUBITOPERATOR_OP_HPP

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "config/config.hpp"
#include "config/type_traits.hpp"

#include "ops/gates/details/coeff_policy.hpp"
#include "ops/gates/details/qubit_operator_term_policy.hpp"
#include "ops/gates/terms_operator_base.hpp"
#include "ops/gates/types.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops {
class QubitOperatorBase {};  // Empty base class

//! Definition of a qubit operator; a sum of terms acting on qubits.
/*!
 *  A term is an operator acting on n qubits and can be represented as:
 *      coefficient * local_operator[0] x ... x local_operator[n-1]
 *  where x is the tensor product. A local operator is a Pauli operator ('I', 'X', 'Y', or 'Z') which acts on one
 *  qubit. In mathematical notation a QubitOperator term is, for example, 0.5 * 'X1 X5', which means that a Pauli X
 *  operator acts on qubit 1 and 5, while the identity operator acts on all the rest qubits.
 *
 *  Note that a Hamiltonian composed of QubitOperators should be a hermitian operator, thus requires the coefficients of
 *  all terms must be real.
 *
 *  QubitOperator has the following attributes set as follows: operators = ('X', 'Y', 'Z'), different_indices_commute =
 *  True.
 */
template <typename coeff_t>
class QubitOperator
    : public TermsOperatorBase<QubitOperator, coeff_t, details::QubitOperatorTermPolicy>
    , public QubitOperatorBase {
 public:
    using base_t = TermsOperatorBase<QubitOperator, coeff_t, details::QubitOperatorTermPolicy>;
    using base_t::get_terms;
    using base_t::is_real_valued;
    using base_t::new_derived_t;
    using base_t::num_targets;
    using base_t::subs;
    using typename base_t::coeff_policy_t;
    using typename base_t::coeff_term_dict_t;
    using typename base_t::coefficient_real_t;
    using typename base_t::coefficient_t;
    using typename base_t::derived_cmplx_t;
    using typename base_t::term_policy_t;
    using self_t = QubitOperator<coefficient_t>;

    using matrix_coeff_t = traits::to_cmplx_type_t<typename coeff_policy_t::matrix_coeff_t>;
    using matrix_t = Eigen::Matrix<matrix_coeff_t, Eigen::Dynamic, Eigen::Dynamic>;
    using sparse_matrix_t = types::csr_matrix_t<matrix_coeff_t>;

    // -------------------------------------------------------------------

    enum class Op : uint8_t { X, Y, Z };
    using op_matrix_t = Eigen::Map<const Eigen::Matrix<std::complex<double>, 2, 2>>;

    static op_matrix_t get_op_matrix(Op op_type);

    // -------------------------------------------------------------------

    static derived_cmplx_t simplify(const self_t& qubit_op);

    // -------------------------------------------------------------------

    using TermsOperatorBase<QubitOperator, coeff_t, details::QubitOperatorTermPolicy>::TermsOperatorBase;
    QubitOperator() = default;
    QubitOperator(const QubitOperator&) = default;
    QubitOperator(QubitOperator&&) = default;
    QubitOperator& operator=(const QubitOperator&) = default;
    QubitOperator& operator=(QubitOperator&&) = default;
    ~QubitOperator() noexcept = default;

    // -------------------------------------------------------------------

    //! Count the number of gates that make up a qubit operator
    MQ_NODISCARD uint32_t count_gates() const noexcept;

    //! Return the matrix representing a QubitOperator
    MQ_NODISCARD std::optional<matrix_t> matrix(/* n_qubits = std::nullopt */) const;

    //! Return the sparse matrix representing a QubitOperator
    MQ_NODISCARD std::optional<sparse_matrix_t> sparse_matrix(std::optional<uint32_t> n_qubits = std::nullopt) const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS
};
}  // namespace mindquantum::ops

namespace mindquantum::traits {
template <typename float_t>
struct to_real_type<ops::QubitOperator<float_t>> {
    using type = ops::QubitOperator<to_real_type_t<float_t>>;
};
template <typename float_t>
struct to_cmplx_type<ops::QubitOperator<float_t>> {
    using type = ops::QubitOperator<to_cmplx_type_t<float_t>>;
};
}  // namespace mindquantum::traits

#include "qubit_operator.tpp"  // NOLINT

#endif /* QUBITOPERATOR_OP_HPP */
