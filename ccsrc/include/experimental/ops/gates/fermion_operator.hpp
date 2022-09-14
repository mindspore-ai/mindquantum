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
#include "experimental/ops/gates/details/coeff_policy.hpp"
#include "experimental/ops/gates/details/fermion_operator_term_policy.hpp"
#include "experimental/ops/gates/terms_operator_base.hpp"
#include "experimental/ops/gates/types.hpp"

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
template <typename coeff_t>
class FermionOperator : public TermsOperatorBase<FermionOperator, coeff_t, details::FermionOperatorTermPolicy> {
 public:
    using base_t = TermsOperatorBase<FermionOperator, coeff_t, details::FermionOperatorTermPolicy>;
    using base_t::get_terms;
    using base_t::new_derived_t;
    using base_t::num_targets;
    using base_t::subs;
    using typename base_t::coeff_policy_t;
    using typename base_t::coeff_term_dict_t;
    using typename base_t::coefficient_real_t;
    using typename base_t::coefficient_t;
    using typename base_t::matrix_coeff_t;
    using typename base_t::term_policy_t;
    using self_t = FermionOperator<coefficient_t>;

    using matrix_t = types::csr_matrix_t<matrix_coeff_t>;

    using TermsOperatorBase<FermionOperator, coeff_t, details::FermionOperatorTermPolicy>::TermsOperatorBase;
    FermionOperator() = default;
    FermionOperator(const FermionOperator&) = default;
    FermionOperator(FermionOperator&&) noexcept(false) = default;
    FermionOperator& operator=(const FermionOperator&) = default;
    FermionOperator& operator=(FermionOperator&&) noexcept(false) = default;
    ~FermionOperator() noexcept = default;

    // -------------------------------------------------------------------

    //! Return the matrix representing a FermionOperator
    MQ_NODISCARD std::optional<matrix_t> matrix(std::optional<uint32_t> n_qubits) const;

    //! Return the normal ordered form of the Fermion Operator.
    MQ_NODISCARD self_t normal_ordered() const;

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    //! Put a term from a FermionOperator into normal ordered form
    /*!
     * \param terms A list of local operators
     * \note Normal ordered form is with high index and creation operator in front.
     */
    static self_t normal_ordered_term_(terms_t local_ops, coefficient_t coeff);
};
}  // namespace mindquantum::ops

#include "experimental/ops/gates/fermion_operator.tpp"

#endif /* FERMION_OPERATOR_HPP */
