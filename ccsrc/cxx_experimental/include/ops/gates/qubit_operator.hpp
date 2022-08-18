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
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "core/config.hpp"

#include "ops/gates/details/qubit_operator_term_policy.hpp"
#include "ops/gates/terms_operator.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops {
class QubitOperatorPR;
constexpr std::tuple<std::complex<double>, TermValue> pauli_products(const TermValue& left_op,
                                                                     const TermValue& right_op);

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
class QubitOperator : public TermsOperator<QubitOperator, details::QubitOperatorTermPolicy> {
    friend TermsOperator<QubitOperator, details::QubitOperatorTermPolicy>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
    using TermsOperator<QubitOperator, details::QubitOperatorTermPolicy>::operator==;

    static constexpr std::string_view kind() {
        return "mindquantum.qubitoperator";
    }

    using TermsOperator::TermsOperator;
    QubitOperator() = default;
    QubitOperator(const QubitOperator&) = default;
    QubitOperator(QubitOperator&&) noexcept = default;
    QubitOperator& operator=(const QubitOperator&) = default;
    QubitOperator& operator=(QubitOperator&&) noexcept = default;
    ~QubitOperator() noexcept = default;

    // -------------------------------------------------------------------

    //! Count the number of gates that make up a qubit operator
    MQ_NODISCARD uint32_t count_gates() const noexcept;

    //! Return the matrix representing a QubitOperator
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    // TODO(dnguyen): Move this into term_policy_t class
    //! Simplify the list of local operators by using commutation and anti-commutation relations
    static std::tuple<terms_t, coefficient_t> simplify_(std::vector<term_t> terms, coefficient_t coeff = 1.);

    // TODO(dnguyen): Move this into term_policy_t class
    //! Simplify the list of local operators by using commutation and anti-commutation relations
    static std::tuple<terms_t, coefficient_t> simplify_(std::vector<py_term_t> py_terms, coefficient_t coeff = 1.);

    // TODO(dnguyen): Move this into term_policy_t class
    //! Sort a list of local operators
    /*!
     * \param local_ops A list of local operators
     * \param coeff A coefficient
     * \note Potentially called by the TermsOperator constructor
     */
    static std::pair<terms_t, coefficient_t> sort_terms_(terms_t local_ops, coefficient_t coeff);
};
}  // namespace mindquantum::ops

#endif /* QUBITOPERATOR_OP_HPP */
