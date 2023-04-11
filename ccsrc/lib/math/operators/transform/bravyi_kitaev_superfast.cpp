//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/transform.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/ops/memory_operator.hpp"

namespace operators::transform {
qubit_op_t bravyi_kitaev_superfast(const fermion_op_t& ops) {
    // ? ops.normal_ordered() does not seem to work
    fermion_op_t fermion_operator = ops;
    edge_matrix_t edge_matrix = get_edge_matrix(fermion_operator);
    edge_enum_t edge_enum = enumerate_edges(edge_matrix);
    // ? auto transf_op = qubit_t(terms_t{}, fermion_operator.constant);
    auto transf_op = qubit_op_t();
    std::set<fermion_op_t::terms_t> transformed_terms;
    for (const auto& [term, coeff] : ops.get_terms()) {
        // Check whether term is already transformed
        if (!transformed_terms.count(term)) {
            std::vector<int> at, a;
            std::unordered_set<int> u;
            for (const auto& [idx, value] : fermion_op_t::terms_t{term.begin(), term.begin() + term.size() / 2}) {
                at.emplace_back(idx);
                u.emplace(idx);
            }
            for (const auto& [idx, value] : fermion_op_t::terms_t{term.begin() + term.size() / 2, term.end()}) {
                a.emplace_back(idx);
                u.emplace(idx);
            }
            // Second term in pair to transform
            fermion_op_t::terms_t term_2;
            std::transform(a.begin(), a.end(), std::back_inserter(term_2), [](int val) {
                return fermion_op_t::term_t{val, fermion::TermValue::Ad};
            });
            std::transform(at.begin(), at.end(), std::back_inserter(term_2), [](int val) {
                return fermion_op_t::term_t{val, fermion::TermValue::A};
            });
            // Check equality between numbers of creation and annihilation operators in term
            if (a.size() != at.size()) {
                throw "Terms in hamiltonian must consist f pairs of creation/annihilation operators";
            }
            // Check whether fermion operator is hermitian
            if ((fermion_operator.get_coeff(term) != fermion_operator.get_coeff(term_2))
                && (fermion_operator.get_coeff(term) != -fermion_operator.get_coeff(term_2))) {
                throw "Fermion operator must be hermitian.";
            }
            // Case of a^i a_j
            if (at.size() == 1) {
                // Case of number operator (i = j)
                if (u.size() == 1) {
                    auto i = at[0];
                    transf_op += transformed_number_operator(i, edge_matrix, edge_enum)
                                 * qubit_op_t("", fermion_operator.get_coeff(term));
                } else {
                    // Case of excitation operator
                    auto i = at[0], j = a[0];  // TODO(xusheng) check with OpenFermion
                    transf_op += transformed_excitation_operator(i, j, edge_matrix, edge_enum)
                                 * qubit_op_t("", fermion_operator.get_coeff(term));
                }
            } else if (at.size() == 2) {
                // Case of a^i a^j a_k a_l
                // Case of Coulomb/exchange operator a^i a_i a^j a_j
                if (u.size() == 2) {
                    auto i = at[0], j = at[1];
                    transf_op += (transformed_exchange_operator(i, j, edge_matrix, edge_enum)
                                  * qubit_op_t("", fermion_operator.get_coeff(term)))
                                 * tensor::ops::init_with_value(-1.0);
                    // -1 factor because of normal ordering (a^i a^j a_i a_j, for i > j)
                } else if (u.size() == 3) {
                    // Case of number excitation operator (a^i a^j a_j a_k)
                    auto i = at[0], j = at[1], k = a[0];
                    if (i == k) {
                        k = a[1];
                        std::swap(i, j);
                    } else if (i == a[1]) {
                        std::swap(i, j);
                    } else if (j == k) {
                        k = a[1];
                    }
                    transf_op += (transformed_number_excitation_operator(i, j, k, edge_matrix, edge_enum)
                                  * qubit_op_t("", fermion_operator.get_coeff(term))
                                  * tensor::ops::init_with_value(std::pow(-1, ((i > j) ^ (j > k)))));
                } else if (u.size() == 4) {
                    // Case of double excitation operator
                    auto i = at[0], j = at[1], k = a[0], l = a[1];
                    transf_op += transformed_double_excitation_operator(i, j, k, l, edge_matrix, edge_enum)
                                 * qubit_op_t("", fermion_operator.get_coeff(term));
                }
            }
            transformed_terms.emplace(term);
            transformed_terms.emplace(term_2);
        }
    }
    return transf_op;
}

edge_matrix_t get_edge_matrix(const fermion_op_t& ops) {
    /*
    Return antisymmetric adjacency matrix (Edge matrix) for graph based on fermion operator for BKSF transform.
    */
    edge_set_t edge_set;
    for (const auto& [term, coeff] : ops.get_terms()) {
        if (term.size() % 2 == 1) {
            throw "Terms in hamiltonian must consist f pairs of creation/annihilation operators";
        }
        std::unordered_map<int, std::vector<int>> a;
        a[0] = std::vector<int>{};
        a[1] = std::vector<int>{};
        for (const auto& [idx, value] : term) {
            int i;
            if (value == fermion::TermValue::Ad) {
                i = 0;
            } else {
                i = 1;
            }
            auto iter = std::find(a[i].begin(), a[i].end(), idx);
            if (iter != a[i].end()) {
                a[i].erase(iter);
            } else {
                a[i ^ 1].emplace_back(idx);
            }
        }
        if (a[1].size() == 2) {
            edge_set.emplace(a[1][0], a[1][1]);
            edge_set.emplace(a[0][0], a[0][1]);
        } else if (a[1].size() == 1) {
            if (a[1][0] > a[0][0]) {
                edge_set.emplace(a[1][0], a[0][0]);
            } else {
                edge_set.emplace(a[0][0], a[1][0]);
            }
        }
    }
    int d = ops.count_qubits();
    edge_matrix_t edge_matrix(d, std::vector<int>(d, 0));
    for (const auto& [i, j] : edge_set) {
        edge_matrix[i][j] = -1;
        edge_matrix[j][i] = 1;
    }
    return edge_matrix;
}

edge_enum_t enumerate_edges(const edge_matrix_t& edge_matrix) {
    int d = edge_matrix.size();
    edge_enum_t edge_enum;
    int n = 0;
    for (int i = 0; i < d; i++) {
        for (int j = i + 1; j < d; j++) {
            if (edge_matrix[i][j] > 0) {
                edge_enum[std::pair<int, int>{i, j}] = n;  // not sure if it is correct
                edge_enum[std::pair<int, int>{j, i}] = n;  // not sure if it is correct
                n += 1;
            }
        }
    }
    return edge_enum;
}

qubit_op_t get_b(int i, const edge_matrix_t& edge_matrix, edge_enum_t& edge_enum) {
    qubit_op_t::terms_t terms;
    for (int j = 0; j < edge_matrix[i].size(); j++) {
        if (edge_matrix[i][j] != 0) {
            terms.emplace_back(edge_enum[std::pair<int, int>{i, j}],
                               qubit::TermValue::Z);  // not sure if it is correct
        }
    }
    auto out = qubit_op_t(terms);
    return out;
}

qubit_op_t get_a(int i, int j, const edge_matrix_t& edge_matrix, edge_enum_t& edge_enum) {
    qubit_op_t::terms_t terms;
    terms.emplace_back(edge_enum[std::pair<int, int>{i, j}], qubit::TermValue::X);
    for (int k = 0; k < j; k++) {
        if (edge_matrix[k][i] != 0) {
            terms.emplace_back(edge_enum[std::pair<int, int>{k, i}],
                               qubit::TermValue::Z);  // not sure if it is correct
        }
    }
    for (int s = 0; s < i; s++) {
        if (edge_matrix[s][j] != 0) {
            terms.emplace_back(edge_enum[std::pair<int, int>{s, j}],
                               qubit::TermValue::Z);  // not sure if it is correct
        }
    }
    auto out = qubit_op_t(terms) * (tensor::ops::init_with_value(1.0 * edge_matrix[i][j]));
    return out;
}

qubit_op_t transformed_number_operator(int i, const edge_matrix_t& edge_matrix, edge_enum_t& edge_enum) {
    return (qubit_op_t("") - get_b(i, edge_matrix, edge_enum)) * tensor::ops::init_with_value(0.5);
}

qubit_op_t transformed_excitation_operator(int i, int j, const edge_matrix_t& edge_matrix, edge_enum_t& edge_enum) {
    qubit_op_t a_ij = get_a(i, j, edge_matrix, edge_enum);
    return (a_ij * get_b(j, edge_matrix, edge_enum) + get_b(i, edge_matrix, edge_enum) * a_ij)
           * tensor::ops::init_with_value(std::complex<float>(0, -0.5));
}

qubit_op_t transformed_exchange_operator(int i, int j, const edge_matrix_t& edge_matrix, edge_enum_t& edge_enum) {
    return ((qubit_op_t("") - get_b(i, edge_matrix, edge_enum)) * (qubit_op_t("") - get_b(j, edge_matrix, edge_enum))
            * tensor::ops::init_with_value(0.25));
}

qubit_op_t transformed_number_excitation_operator(int i, int j, int k, const edge_matrix_t& edge_matrix,
                                                  edge_enum_t& edge_enum) {
    auto a_ik = get_a(i, k, edge_matrix, edge_enum);
    return ((a_ik * get_b(k, edge_matrix, edge_enum) + get_b(i, edge_matrix, edge_enum) * a_ik)
            * (qubit_op_t("") - get_b(j, edge_matrix, edge_enum)) * tensor::ops::init_with_value(-0.25));
}

qubit_op_t transformed_double_excitation_operator(int i, int j, int k, int l, const edge_matrix_t& edge_matrix,
                                                  edge_enum_t& edge_enum) {
    auto b_i = get_b(i, edge_matrix, edge_enum);
    auto b_j = get_b(j, edge_matrix, edge_enum);
    auto b_k = get_b(k, edge_matrix, edge_enum);
    auto b_l = get_b(l, edge_matrix, edge_enum);
    return (get_a(i, j, edge_matrix, edge_enum) * get_a(k, l, edge_matrix, edge_enum)
            * (qubit_op_t("", parameter::ParameterResolver(tensor::ops::init_with_value(-1.0)))
               - (b_i * b_j + b_k * b_l) + b_i * b_k + b_i * b_l + b_j * b_k + b_j * b_l + b_i * b_j * b_k * b_l)
            * tensor::ops::init_with_value(0.125));
}
}  // namespace operators::transform
