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

#ifndef JORDAN_WIGNER_TRANSFORM_TPP
#define JORDAN_WIGNER_TRANSFORM_TPP

#include <lru_cache/lru_cache.h>

#include "core/parameter_resolver.hpp"
#include "ops/transform/fermion_number_operator.hpp"
#include "ops/transform/jordan_wigner.hpp"
#include "ops/transform/transform_ladder_operator.hpp"
#include "ops/transform/types.hpp"

namespace lru_cache {
template <typename coeff_t>
using qo_t = mindquantum::ops::QubitOperator<coeff_t>;

// -----------------------------------------------------------------------------

template <typename key_t, typename value_t, std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptionsBoostBase {
    using index_t = internal::index_type_for<cache_size>;

    static constexpr index_t MAX_SIZE = std::numeric_limits<index_t>::max() - 1;
    static_assert(cache_size <= MAX_SIZE);

    using Key = key_t;
    using Value = value_t;
    using IndexType = index_t;

    using Map = std::unordered_map<key_t, index_t, boost::hash<key_t>>;

    using NodeContainer = ArrayNodeContainer<internal::Node<key_t, value_t, index_t>, cache_size>;

    static constexpr bool ByAccessOrder = by_access_order;
};

template <std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptions<mindquantum::ops::term_t, qo_t<std::complex<double>>, cache_size, by_access_order>
    : public StaticLruCacheOptionsBoostBase<mindquantum::ops::term_t, qo_t<std::complex<double>>, cache_size,
                                            by_access_order> {};
template <std::size_t cache_size, bool by_access_order>
struct StaticLruCacheOptions<mindquantum::ops::term_t, qo_t<mindquantum::ParameterResolver<std::complex<double>>>,
                             cache_size, by_access_order>
    : public StaticLruCacheOptionsBoostBase<mindquantum::ops::term_t,
                                            qo_t<mindquantum::ParameterResolver<std::complex<double>>>, cache_size,
                                            by_access_order> {};
}  // namespace lru_cache

namespace mindquantum::ops::transform {
template <typename fermion_op_t>
auto jordan_wigner(const fermion_op_t& ops) -> to_qubit_operator_t<traits::to_cmplx_type_t<fermion_op_t>> {
    using qubit_op_t = to_qubit_operator_t<traits::to_cmplx_type_t<fermion_op_t>>;
    using coefficient_t = typename qubit_op_t::coefficient_t;

    static constexpr auto cache_size_ = 1000UL;
    static auto cache_ = lru_cache::staticc::make_cache<cache_size_, term_t, qubit_op_t>();

    auto transf_op = qubit_op_t();
    for (const auto& [terms, coeff] : ops.get_terms()) {
        auto transformed_term = qubit_op_t(terms_t{}, static_cast<coefficient_t>(coeff));
        // auto transformed_term = qubit_op_t::identity() * coeff;
        for (const auto& term : terms) {
            if (auto cached_mult = cache_.get_or_null(term); cached_mult != nullptr) {
                transformed_term *= *cached_mult;
            } else {
                const auto& [idx, value] = term;
                qlist_t z(idx);
                std::iota(begin(z), end(z), 0UL);
                transformed_term *= cache_.insert(
                    term, transform_ladder_operator<fermion_op_t>(value, {idx}, {}, z, {}, {idx}, z));
            }
        }
        transf_op += transformed_term;
    }
    return transf_op;
}

template <typename qubit_op_t>
auto reverse_jordan_wigner(const qubit_op_t& ops, int n_qubits)
    -> to_fermion_operator_t<traits::to_cmplx_type_t<qubit_op_t>> {
    using fermion_op_t = to_fermion_operator_t<traits::to_cmplx_type_t<qubit_op_t>>;
    using coefficient_t = typename fermion_op_t::coefficient_t;
    auto local_n_qubits = ops.count_qubits();
    if (n_qubits <= 0) {
        n_qubits = local_n_qubits;
    }
    if (n_qubits < local_n_qubits) {
        throw std::runtime_error("Target qubits number is less than local qubits of operator.");
    }
    auto transf_op = fermion_op_t();
    for (const auto& [term, coeff] : ops.get_terms()) {
        auto transformed_term = fermion_op_t("", fermion_op_t::coeff_policy_t::one);
        if (term.size() != 0) {
            auto working_term = traits::to_cmplx_type_t<qubit_op_t>(term);
            auto pauli_operator = term[term.size() - 1];
            bool finished = false;
            while (!finished) {
                auto& [idx, value] = pauli_operator;
                fermion_op_t trans_pauli;
                if (value == TermValue::Z) {
                    trans_pauli = fermion_op_t("", fermion_op_t::coeff_policy_t::one);
                    trans_pauli += fermion_number_operator<fermion_op_t>(n_qubits, idx, coefficient_t(-2.0));
                } else {
                    auto raising_term = fermion_op_t({idx, TermValue::adg});
                    auto lowering_term = fermion_op_t({idx, TermValue::a});
                    if (value == TermValue::Y) {
                        raising_term *= coefficient_t(std::complex<double>(0, 1));
                        lowering_term *= coefficient_t(std::complex<double>(0, -1));
                    }
                    trans_pauli += raising_term;
                    trans_pauli += lowering_term;
                    for (auto j = 0; j < idx; j++) {
                        working_term = qubit_op_t({idx - 1 - j, TermValue::Z}) * working_term;
                    }
                    auto s_coeff = working_term.singlet_coeff();
                    trans_pauli *= s_coeff;
                    working_term *= (fermion_op_t::coeff_policy_t::one / s_coeff);
                }
                int working_qubit = static_cast<int>(idx) - 1;
                for (auto [local_term, local_coeff] : working_term.get_terms()) {
                    for (auto it = local_term.rbegin(); it != local_term.rend(); it++) {
                        const auto& [local_idx, local_value] = *it;
                        if (static_cast<int>(local_idx) <= working_qubit) {
                            pauli_operator = term_t({local_idx, local_value});
                            finished = false;
                            break;
                        } else {
                            finished = true;
                        }
                    }
                    break;
                }
                transformed_term *= trans_pauli;
            }
        }
        transformed_term *= coeff;
        transf_op += transformed_term;
    }
    return transf_op;
}
}  // namespace mindquantum::ops::transform

#endif /* JORDAN_WIGNER_TRANSFORM_TPP */
