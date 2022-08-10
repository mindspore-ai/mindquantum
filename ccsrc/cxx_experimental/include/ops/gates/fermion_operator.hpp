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

#ifndef FERMION_OPERATOR_HPP
#define FERMION_OPERATOR_HPP

#include <complex>
#include <cstdint>
#include <string>
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

//! Definition of a fermionic operator
/*!
 *
 *  The Fermion Operator such as FermionOperator(' 4^ 3 9 3^ ') are used to represent
 *  \f$a_4^\dagger a_3 a_9 a_3^\dagger\f$.
 *
 *  These are the Basic Operators to describe a fermionic system, such as a Molecular system. The FermionOperator are
 *  follows the anti-commutation relationship.
 */
class FermionOperator : public TermsOperator<FermionOperator> {
    friend TermsOperator<FermionOperator>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
    using TermsOperator<FermionOperator>::operator==;

    static constexpr std::string_view kind() {
        return "mindquantum.fermionoperator";
    }

    using TermsOperator::TermsOperator;
    FermionOperator() = default;
    FermionOperator(const FermionOperator&) = default;
    FermionOperator(FermionOperator&&) noexcept = default;
    FermionOperator& operator=(const FermionOperator&) = default;
    FermionOperator& operator=(FermionOperator&&) noexcept = default;
    ~FermionOperator() noexcept = default;

    //! Constructor from a string representing a list of terms
    /*!
     * \note If parsing the string fails, the resulting QubitOperator object will represent the identity. If logging is
     *       enabled, an error message will be printed inside the log with an appropriate error message.
     */
    explicit FermionOperator(std::string_view terms_string, coefficient_t coeff = 1.0);

    // -------------------------------------------------------------------

    //! Return the matrix representing a FermionOperator
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Split the operator into its individual components
    MQ_NODISCARD std::vector<FermionOperator> split() const noexcept;

    // NB: This makes little sense since the ordering is given by the storage (std::map or else)
    // //! Return the normal ordered form of the Fermion Operator.
    // MQ_NODISCARD FermionOperator normal_ordered() const;

    //! Convert a FermionOperatoir to a string
    MQ_NODISCARD std::string to_string() const noexcept;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    //! Return the normal ordered term of the FermionOperator with high index and creation operator in front.
    static FermionOperator normal_ordered_term_(terms_t terms, coefficient_t coeff);

    //! Convert a string of space-separated fermion operators into an array of terms
    static terms_t parse_string_(std::string_view terms_string);

    //! Simplify the list of local operators
    static std::tuple<std::vector<term_t>, coefficient_t> simplify_(terms_t terms, coefficient_t coeff = 1.);
};

}  // namespace mindquantum::ops

#endif /* FERMION_OPERATOR_HPP */
