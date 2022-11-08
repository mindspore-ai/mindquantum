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

#ifndef FERMION_NUMBER_OPERATOR_HPP
#define FERMION_NUMBER_OPERATOR_HPP

#include "config/config.hpp"

#include "ops/transform/types.hpp"

namespace mindquantum::ops::transform {
//! Jordan Wigner transform that transform a Fermion operator to qubit operator.
template <typename fermion_op_t>
MQ_NODISCARD fermion_op_t fermion_number_operator(int n_modes, int mode = -1,
                                                  typename fermion_op_t::coefficient_t coeff
                                                  = fermion_op_t::coeff_policy_t::one);
}  // namespace mindquantum::ops::transform

#include "fermion_number_operator.tpp"  // NOLINT

#endif
