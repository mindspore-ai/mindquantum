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
#include <string_view>
#include <tuple>
#include <vector>

#include <Eigen/SparseCore>

#include "core/config.hpp"

#include "ops/gates/terms_operator.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops {

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
class QubitOperator : public TermsOperator<QubitOperator> {
    friend TermsOperator<QubitOperator>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;

    static constexpr std::string_view kind() {
        return "mindquantum.qubitoperator";
    }

    QubitOperator() = default;

    explicit QubitOperator(term_t term, coefficient_t coefficient = 1.0);

    explicit QubitOperator(const terms_t& term, coefficient_t coeff = 1.0);

    explicit QubitOperator(complex_term_dict_t terms);

    explicit QubitOperator(std::string_view terms_string);

    // -------------------------------------------------------------------

    MQ_NODISCARD uint32_t count_gates() const noexcept;

    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Split the operator into its individual components
    MQ_NODISCARD std::vector<QubitOperator> split() const noexcept;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    //! Convert a string of space-separated qubit operators into an array of terms
    static terms_t parse_string_(std::string_view terms_string);

    //! Simplify the list of local operators by using commutation and anti-commutation relations
    static std::tuple<terms_t, coefficient_t> simplify_(std::vector<term_t> terms, coefficient_t coeff = 1.);
};
}  // namespace mindquantum::ops

#endif /* QUBITOPERATOR_OP_HPP */
