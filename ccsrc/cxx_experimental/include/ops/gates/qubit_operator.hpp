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

#include <algorithm>
#include <complex>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "ops/meta/dagger.hpp"

namespace mindquantum::ops {
class QubitOperator {
    std::map<std::pair<char, char>, std::pair<std::complex<double>, char>> PAULI_OPERATOR_PRODUCTS_ = {
        {{'I', 'I'}, {1., 'I'}},       {{'I', 'X'}, {1., 'X'}},      {{'X', 'I'}, {1., 'X'}},
        {{'I', 'Y'}, {1., 'Y'}},       {{'Y', 'I'}, {1., 'Y'}},      {{'I', 'Z'}, {1., 'Z'}},
        {{'Z', 'I'}, {1., 'Z'}},       {{'X', 'X'}, {1., 'I'}},      {{'Y', 'Y'}, {1., 'I'}},
        {{'Z', 'Z'}, {1., 'I'}},       {{'X', 'Y'}, {{0, 1.}, 'Z'}}, {{'X', 'Z'}, {{0, -1.}, 'Y'}},
        {{'Y', 'X'}, {{0, -1.}, 'Z'}}, {{'Y', 'Z'}, {{0, 1.}, 'X'}}, {{'Z', 'X'}, {{0, 1.}, 'Y'}},
        {{'Z', 'Y'}, {{0, -1.}, 'X'}}};

 public:
    using non_const_num_targets = void;

    using ComplexTerm = std::pair<std::vector<std::pair<uint32_t, char>>, std::complex<double>>;
    using ComplexTermsDict = std::map<std::vector<std::pair<uint32_t, char>>, std::complex<double>>;

    static constexpr std::string_view kind() {
        return "projectq.qubitoperator";
    }

    explicit QubitOperator(uint32_t num_targets, ComplexTermsDict terms)
        : num_targets_(num_targets), terms_(std::move(terms)) {
    }

    td::Operator adjoint() const {
        return DaggerOperation(*this);
    }

    uint32_t num_targets() const {
        return num_targets_;
    }

    bool operator==(const QubitOperator& other) const {
        return is_close(other);
    }

    // -------------------------------------------------------------------

    const ComplexTermsDict& get_terms() const {
        return terms_;
    }

    QubitOperator Subtract(QubitOperator const& other) const {
        ComplexTermsDict result_terms = terms_;
        for (auto& [term, other_coefficient] : other.get_terms()) {
            if (result_terms.count(term)) {
                if (abs(result_terms.at(term) - other_coefficient) > 0) {
                    result_terms.at(term) -= other_coefficient;
                } else {
                    result_terms.erase(term);
                }
            } else {
                result_terms[term] = -other_coefficient;
            }
        }
        return QubitOperator(num_targets_, result_terms);
    }

    QubitOperator operator-(QubitOperator const& other) const {
        return Subtract(other);
    }

    QubitOperator Multiply(QubitOperator const& other) const {
        ComplexTermsDict result_terms;
        for (auto& [left_term, coefficient] : terms_) {
            for (auto& [right_term, other_coefficient] : other.get_terms()) {
                auto new_coefficient = coefficient * other_coefficient;

                // Loop through local operators and create new sorted
                // list of representing the product local operator:
                std::vector<std::pair<uint32_t, char>> product_operators;
                unsigned left_operator_index = 0;
                unsigned right_operator_index = 0;
                unsigned n_operators_left = std::size(left_term);
                unsigned n_operators_right = std::size(right_term);
                while (left_operator_index < n_operators_left && right_operator_index < n_operators_right) {
                    auto [left_qubit, left_loc_op] = left_term[left_operator_index];
                    auto [right_qubit, right_loc_op] = right_term[right_operator_index];

                    // Multiply local operators acting on the same
                    // qubit
                    if (left_qubit == right_qubit) {
                        left_operator_index += 1;
                        right_operator_index += 1;
                        auto [scalar, loc_op] = PAULI_OPERATOR_PRODUCTS_.at(std::make_pair(left_loc_op, right_loc_op));

                        // Add new term.
                        if (loc_op != 'I') {
                            product_operators.push_back({left_qubit, loc_op});
                            new_coefficient *= scalar;
                        }
                        // Note if loc_op == 'I', then scalar == 1.0
                    }
                    // If left_qubit > right_qubit, add right_loc_op;
                    // else, add left_loc_op.
                    else if (left_qubit > right_qubit) {
                        product_operators.push_back({right_qubit, right_loc_op});
                        right_operator_index += 1;
                    } else {
                        product_operators.push_back({left_qubit, left_loc_op});
                        left_operator_index += 1;
                    }
                }

                // Finish the remaining operators :
                if (left_operator_index == n_operators_left) {
                    for (unsigned index = right_operator_index; index < std::size(right_term); index++) {
                        product_operators.push_back(right_term[index]);
                    }
                } else if (right_operator_index == n_operators_right) {
                    for (unsigned index = left_operator_index; index < std::size(left_term); index++) {
                        product_operators.push_back(left_term[index]);
                    }
                }

                // Add to result dict
                if (result_terms.count(product_operators)) {
                    result_terms[product_operators] += new_coefficient;
                } else {
                    result_terms[product_operators] = new_coefficient;
                }
            }
        }
        return QubitOperator(num_targets_, result_terms);
    }

    QubitOperator operator*(QubitOperator const& other) const {
        return Multiply(other);
    }

    bool is_close(const QubitOperator& other, double rel_tol = 1e-12, double abs_tol = 1e-12) const {
        auto it_this = std::begin(terms_);
        auto it_other = std::begin(other.terms_);

        while (it_this != std::end(terms_) && it_other != std::end(other.terms_)) {
            if (it_this->first < it_other->first) {
                if (std::abs(it_this->second) <= abs_tol) {
                    ++it_this;
                } else {
                    return false;
                }
            } else if (it_other->first < it_this->first) {
                if (std::abs(it_other->second) <= abs_tol) {
                    ++it_other;
                } else {
                    return false;
                }
            } else {
                // Equal keys
                const auto& a = it_this->second;
                const auto& b = it_other->second;

                if (std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol)) {
                    ++it_this;
                    ++it_other;
                } else {
                    return false;
                }
            }
        }

        while (it_this != std::end(terms_)) {
            if (it_this->first < it_other->first) {
                if (std::abs(it_this->second) <= abs_tol) {
                    ++it_this;
                } else {
                    return false;
                }
            }
        }

        while (it_other != std::end(other.terms_)) {
            if (it_other->first < it_this->first) {
                if (std::abs(it_other->second) <= abs_tol) {
                    ++it_other;
                } else {
                    return false;
                }
            }
        }

        return true;
    }

    bool is_identity(double abs_tol = 1e-12) const {
        for (const auto& [term, coeff] : terms_) {
            if (!(std::abs(coeff) <= abs_tol)) {
                return false;
            }
        }
        return true;
    }

 private:
    uint32_t num_targets_;
    ComplexTermsDict terms_;
};
}  // namespace mindquantum::ops

#endif /* QUBITOPERATOR_OP_HPP */
