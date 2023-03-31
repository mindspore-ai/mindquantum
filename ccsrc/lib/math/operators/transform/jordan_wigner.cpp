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
#include "math/operators/transform/jordan_wigner.hpp"

#include <numeric>

#include "math/operators/transform/fermion_number_operator.hpp"
#include "math/operators/transform/transform_ladder_operator.hpp"
#include "math/tensor/ops/memory_operator.hpp"
namespace operators::transform {
qubit_op_t jordan_wigner(const fermion_op_t& ops) {
    // static constexpr auto cache_size_ = 1000UL;

    auto transf_op = qubit_op_t();
    for (const auto& [terms, coeff] : ops.get_terms()) {
        auto transformed_term = qubit_op_t("", coeff);
        // auto transformed_term = qubit_op_t::identity() * coeff;
        for (const auto& term : terms) {
            // if (auto cached_mult = cache_.get_or_null(term); cached_mult != nullptr) {
            //     transformed_term *= *cached_mult;
            // } else {
            const auto& [idx, value] = term;
            std::vector<size_t> z(idx);
            std::iota(begin(z), end(z), 0UL);
            transformed_term *= transform_ladder_operator(value, {idx}, {}, z, {}, {idx}, z);
            // }
        }
        transf_op += transformed_term;
    }
    return transf_op;
}

fermion_op_t reverse_jordan_wigner(const qubit_op_t& ops, int n_qubits) {
    auto local_n_qubits = ops.count_qubits();
    if (n_qubits <= 0) {
        n_qubits = local_n_qubits;
    }
    if (n_qubits < local_n_qubits) {
        throw std::runtime_error("Target qubits number is less than local qubits of operator.");
    }
    auto transf_op = fermion_op_t();
    for (const auto& [term, coeff] : ops.get_terms()) {
        auto transformed_term = fermion_op_t("");
        if (term.size() != 0) {
            auto working_term = term;
            auto pauli_operator = term[term.size() - 1];
            bool finished = false;
            while (!finished) {
                auto& [idx, value] = pauli_operator;
                fermion_op_t trans_pauli;
                if (value == qubit::TermValue::Z) {
                    trans_pauli = fermion_op_t("");
                    trans_pauli += fermion_number_operator(n_qubits, idx, tn::ops::init_with_value(-2.0));
                } else {
                    auto raising_term = fermion_op_t({idx, fermion::TermValue::Ad});
                    auto lowering_term = fermion_op_t({idx, fermion::TermValue::A});
                    if (value == qubit::TermValue::Y) {
                        raising_term *= tn::ops::init_with_value(std::complex<double>(0, 1.0));
                        lowering_term *= tn::ops::init_with_value(std::complex<double>(0, -1.0));
                    }
                    trans_pauli += raising_term;
                    trans_pauli += lowering_term;
                    for (auto j = 0; j < idx; j++) {
                        working_term = qubit_op_t({idx - 1 - j, qubit::TermValue::Z}) * working_term;
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
}  // namespace operators::transform

#endif /* JORDAN_WIGNER_TRANSFORM_TPP */
