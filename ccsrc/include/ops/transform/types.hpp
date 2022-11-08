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

#ifndef OPS_TRANSFORM_TYPES_HPP
#define OPS_TRANSFORM_TYPES_HPP

#include <vector>

#include "config/config.hpp"

#include "ops/gates/fermion_operator.hpp"
#include "ops/gates/qubit_operator.hpp"

// =============================================================================

namespace mindquantum::ops::transform {
template <typename fermion_op_t>
struct to_qubit_operator;

template <typename coeff_t>
struct to_qubit_operator<FermionOperator<coeff_t>> {
    using type = QubitOperator<coeff_t>;
};

template <typename fermion_op_t>
using to_qubit_operator_t = typename to_qubit_operator<fermion_op_t>::type;

// -----------------------------------------------------------------------------

template <typename qubit_op_t>
struct to_fermion_operator;

template <typename coeff_t>
struct to_fermion_operator<QubitOperator<coeff_t>> {
    using type = QubitOperator<coeff_t>;
};

template <typename fermion_op_t>
using to_fermion_operator_t = typename to_fermion_operator<fermion_op_t>::type;

// -----------------------------------------------------------------------------

using qlist_t = std::vector<term_t::first_type>;
}  // namespace mindquantum::ops::transform

// =============================================================================
#endif
