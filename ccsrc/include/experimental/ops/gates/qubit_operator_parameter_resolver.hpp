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

#include "core/parameter_resolver.hpp"

#include "experimental/core/config.hpp"
#include "experimental/ops/gates/details/parameterresolver_coeff_policy.hpp"
#include "experimental/ops/gates/details/qubit_operator_term_policy.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"
#include "experimental/ops/gates/terms_operator.hpp"

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
    QubitOperatorPR(QubitOperatorPR&&) = default;
    QubitOperatorPR& operator=(const QubitOperatorPR&) = default;
    QubitOperatorPR& operator=(QubitOperatorPR&&) = default;
    ~QubitOperatorPR() noexcept = default;

    // -------------------------------------------------------------------

    //! Count the number of gates that make up a qubit operator
    MQ_NODISCARD uint32_t count_gates() const noexcept;

    //! Return the matrix representing a QubitOperator
    MQ_NODISCARD std::optional<csr_matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS
};
}  // namespace mindquantum::ops

#endif /* QUBITOPERATOR_PARAMETER_RESOLVER_HPP */
