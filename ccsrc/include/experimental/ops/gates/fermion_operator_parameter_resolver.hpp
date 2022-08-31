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

#ifndef FERMION_OPERATOR_PARAMETER_RESOLVER_HPP
#define FERMION_OPERATOR_PARAMETER_RESOLVER_HPP

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
#include "experimental/ops/gates/details/parameterresolver_coeff_policy.hpp"
#include "experimental/ops/gates/terms_operator.hpp"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops {
//! Definition of a fermionic operator
/*!
 *
 *  The Fermion Operator such as FermionOperatorPR(' 4^ 3 9 3^ ') are used to represent
 *  \f$a_4^\dagger a_3 a_9 a_3^\dagger\f$.
 *
 *  These are the Basic Operators to describe a fermionic system, such as a Molecular system. The FermionOperatorPR are
 *  follows the anti-commutation relationship.
 */
class FermionOperatorPR
    : public TermsOperator<FermionOperatorPR, details::FermionOperatorTermPolicy, details::CmplxDoublePRCoeffPolicy> {
    friend TermsOperator<FermionOperatorPR, details::FermionOperatorTermPolicy, details::CmplxDoublePRCoeffPolicy>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
    using TermsOperator<FermionOperatorPR, details::FermionOperatorTermPolicy,
                        details::CmplxDoublePRCoeffPolicy>::operator==;

    static constexpr std::string_view kind() {
        return "mindquantum.fermionoperator_pr";
    }

    using TermsOperator::TermsOperator;
    FermionOperatorPR() = default;
    FermionOperatorPR(const FermionOperatorPR&) = default;
    FermionOperatorPR(FermionOperatorPR&&) noexcept = default;
    FermionOperatorPR& operator=(const FermionOperatorPR&) = default;
    FermionOperatorPR& operator=(FermionOperatorPR&&) noexcept = default;
    ~FermionOperatorPR() noexcept = default;

    // -------------------------------------------------------------------

    //! Return the matrix representing a FermionOperatorPR
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Return the normal ordered form of the Fermion Operator.
    MQ_NODISCARD FermionOperatorPR normal_ordered() const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    // TODO(dnguyen): Move this into term_policy_t class?
    //! Put a term from a FermionOperatorPR into normal ordered form
    /*!
     * \param terms A list of local operators
     * \note Normal ordered form is with high index and creation operator in front.
     */
    static FermionOperatorPR normal_ordered_term_(terms_t local_ops, coefficient_t coeff);

    // TODO(dnguyen): Move this into term_policy_t class
    //! Simplify the list of local operators
    static std::tuple<std::vector<term_t>, coefficient_t> simplify_(terms_t terms,
                                                                    coefficient_t coeff = coeff_policy_t::one);

    // TODO(dnguyen): Move this into term_policy_t class
    //! Simplify the list of local operators
    static std::tuple<std::vector<term_t>, coefficient_t> simplify_(py_terms_t py_terms,
                                                                    coefficient_t coeff = coeff_policy_t::one);

    // TODO(dnguyen): Move this into term_policy_t class
    //! Sort a list of local operators
    /*!
     * \param local_ops A list of local operators
     * \param coeff A coefficient
     * \note Potentially called by the TermsOperator constructor
     * \note Defaults to normal ordering by calling \sa normal_ordered_term_
     */
    static std::pair<terms_t, coefficient_t> sort_terms_(terms_t local_ops, coefficient_t coeff);
};

}  // namespace mindquantum::ops

#endif /* FERMION_OPERATOR_PARAMETER_RESOLVER_HPP */
