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
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "experimental/core/config.hpp"
#include "experimental/ops/gates/details/fermion_operator_term_policy.hpp"
#include "experimental/ops/gates/details/std_complex_coeff_policy.hpp"
#include "experimental/ops/gates/terms_operator.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops::test {}  // namespace mindquantum::ops::test

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
class FermionOperator
    : public TermsOperator<FermionOperator, details::FermionOperatorTermPolicy, details::CmplxDoubleCoeffPolicy> {
    friend TermsOperator<FermionOperator, details::FermionOperatorTermPolicy, details::CmplxDoubleCoeffPolicy>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
    using TermsOperator<FermionOperator, details::FermionOperatorTermPolicy,
                        details::CmplxDoubleCoeffPolicy>::operator==;

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

    // -------------------------------------------------------------------

    //! Return the matrix representing a FermionOperator
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Return the normal ordered form of the Fermion Operator.
    MQ_NODISCARD FermionOperator normal_ordered() const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    //! Put a term from a FermionOperator into normal ordered form
    /*!
     * \param terms A list of local operators
     * \note Normal ordered form is with high index and creation operator in front.
     */
    static FermionOperator normal_ordered_term_(terms_t local_ops, coefficient_t coeff);
};

}  // namespace mindquantum::ops

#endif /* FERMION_OPERATOR_HPP */
