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

#ifndef QUBITOPERATOR_PARAMETER_RESOLVER_HPP
#define QUBITOPERATOR_PARAMETER_RESOLVER_HPP

#include <complex>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "core/config.hpp"

#include "ops/gates/details/parameterresolver_coeff_policy.hpp"
#include "ops/gates/details/qubit_operator_term_policy.hpp"
#include "ops/gates/qubit_operator.hpp"
#include "ops/gates/terms_operator.hpp"
#include "pr/parameter_resolver.h"

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum::ops {
//! Definition of a qubit operator (PR); a sum of terms acting on qubits.
class QubitOperatorPR
    : public TermsOperator<QubitOperatorPR, details::QubitOperatorTermPolicy, details::CmplxDoublePRCoeffPolicy> {
    friend TermsOperator<QubitOperatorPR, details::QubitOperatorTermPolicy, details::CmplxDoublePRCoeffPolicy>;

 public:
    using csr_matrix_t = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
    using TermsOperator<QubitOperatorPR, details::QubitOperatorTermPolicy,
                        details::CmplxDoublePRCoeffPolicy>::operator==;

    static constexpr std::string_view kind() {
        return "mindquantum.qubitoperator_pr";
    }

    using TermsOperator::TermsOperator;
    QubitOperatorPR() = default;
    QubitOperatorPR(const QubitOperatorPR&) = default;
    QubitOperatorPR(QubitOperatorPR&&) noexcept = default;
    QubitOperatorPR& operator=(const QubitOperatorPR&) = default;
    QubitOperatorPR& operator=(QubitOperatorPR&&) noexcept = default;
    ~QubitOperatorPR() noexcept = default;

    // -------------------------------------------------------------------

    //! Count the number of gates that make up a qubit operator
    MQ_NODISCARD uint32_t count_gates() const noexcept;

    //! Return the matrix representing a QubitOperator
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Set parameter with given params PR
    QubitOperatorPR subs(const coefficient_t& params_pr) noexcept;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    //! Simplify the list of local operators by using commutation and anti-commutation relations
    static std::tuple<terms_t, coefficient_t> simplify_(std::vector<term_t> terms,
                                                        coefficient_t coeff = coeff_policy_t::one);

    //! Simplify the list of local operators by using commutation and anti-commutation relations
    static std::tuple<terms_t, coefficient_t> simplify_(std::vector<py_term_t> py_terms,
                                                        coefficient_t coeff = coeff_policy_t::one);

    //! Sort a list of local operators
    /*!
     * \param local_ops A list of local operators
     * \param coeff A coefficient
     * \note Potentially called by the TermsOperator constructor
     */
    static std::pair<terms_t, coefficient_t> sort_terms_(terms_t local_ops, coefficient_t coeff);
};
}  // namespace mindquantum::ops

#endif /* QUBITOPERATOR_PARAMETER_RESOLVER_HPP */
