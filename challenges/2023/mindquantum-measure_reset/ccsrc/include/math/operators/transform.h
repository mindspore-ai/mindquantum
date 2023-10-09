/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATH_OPERATORS_TRANSFORM
#define MATH_OPERATORS_TRANSFORM
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "math/operators/fermion_operator_view.h"
#include "math/operators/qubit_operator_view.h"

namespace operators::transform {
namespace tn = tensor;
using fermion_op_t = fermion::FermionOperator;
using qubit_op_t = qubit::QubitOperator;
using qlist_t = std::vector<size_t>;
qubit_op_t transform_ladder_operator(const fermion::TermValue& value, const qlist_t& x1, const qlist_t& y1,
                                     const qlist_t& z1, const qlist_t& x2, const qlist_t& y2, const qlist_t& z2);
qubit_op_t jordan_wigner(const fermion_op_t& ops);

// TODO(xusheng): Reverse jordan wigner transform has bug.
fermion_op_t reverse_jordan_wigner(const qubit_op_t& ops, int n_qubits = -1);
qubit_op_t ternary_tree(const fermion_op_t& ops, int n_qubits);
int get_qubit_index(const qlist_t& p, int i);
qubit_op_t parity(const fermion_op_t& ops, int n_qubits = -1);
fermion_op_t fermion_number_operator(int n_modes, int mode = -1,
                                     const parameter::ParameterResolver& coeff
                                     = parameter::ParameterResolver(tn::ops::ones(1)));

// -----------------------------------------------------------------------------

qubit_op_t bravyi_kitaev(const fermion_op_t& ops, int n_qubits = -1);

std::unordered_set<qubit_op_t::term_t::first_type> parity_set(qubit_op_t::term_t::first_type idx);

std::unordered_set<qubit_op_t::term_t::first_type> occupation_set(qubit_op_t::term_t::first_type idx);

std::unordered_set<qubit_op_t::term_t::first_type> update_set(qubit_op_t::term_t::first_type idx, int n_qubits);

// -----------------------------------------------------------------------------

using edge_matrix_t = std::vector<std::vector<int>>;
using edge_enum_t = std::map<std::pair<int, int>, int>;
using edge_set_t = std::set<std::pair<int, int>>;

//! Bravyi kitaev superfast transform that transform a Fermion operator to qubit operator.
edge_matrix_t get_edge_matrix(const fermion_op_t& ops);

edge_enum_t enumerate_edges(const edge_matrix_t& edge_matrix);

qubit_op_t get_b(int i, const edge_matrix_t& edge_matrix, const edge_enum_t& edge_enum);

qubit_op_t get_a(int i, int j, const edge_matrix_t& edge_matrix, const edge_enum_t& edge_enum);

qubit_op_t transformed_number_operator(int i, const edge_matrix_t& edge_matrix, const edge_enum_t& edge_enum);

qubit_op_t transformed_excitation_operator(int i, int j, const edge_matrix_t& edge_matrix,
                                           const edge_enum_t& edge_enum);

qubit_op_t transformed_exchange_operator(int i, int j, const edge_matrix_t& edge_matrix, const edge_enum_t& edge_enum);

qubit_op_t transformed_number_excitation_operator(int i, int j, int k, const edge_matrix_t& edge_matrix,
                                                  const edge_enum_t& edge_enum);

qubit_op_t transformed_double_excitation_operator(int i, int j, int k, int l, const edge_matrix_t& edge_matrix,
                                                  const edge_enum_t& edge_enum);

qubit_op_t bravyi_kitaev_superfast(const fermion_op_t& ops);
}  // namespace operators::transform

#endif /* MATH_OPERATORS_TRANSFORM */
