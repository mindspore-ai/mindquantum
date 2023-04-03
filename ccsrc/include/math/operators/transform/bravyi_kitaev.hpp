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

#ifndef BRAVYI_KITAEV_TRANSFORM_HPP
#define BRAVYI_KITAEV_TRANSFORM_HPP
#include <unordered_set>

#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/qubit_operator_view.hpp"
namespace operators::transform {
using qubit_op_t = qubit::QubitOperator;
using fermion_op_t = fermion::FermionOperator;

qubit_op_t bravyi_kitaev(const fermion_op_t& ops, int n_qubits = -1);

std::unordered_set<qubit_op_t::term_t::first_type> parity_set(qubit_op_t::term_t::first_type idx);

std::unordered_set<qubit_op_t::term_t::first_type> occupation_set(qubit_op_t::term_t::first_type idx);

std::unordered_set<qubit_op_t::term_t::first_type> update_set(qubit_op_t::term_t::first_type idx, int n_qubits);
}  // namespace operators::transform

#endif
