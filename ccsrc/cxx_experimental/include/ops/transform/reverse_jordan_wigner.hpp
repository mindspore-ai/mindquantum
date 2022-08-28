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

#ifndef REVERSE_JORDAN_WIGNER_TRANSFORM_HPP
#define REVERSE_JORDAN_WIGNER_TRANSFORM_HPP

#include "ops/transform/types.hpp"

namespace mindquantum::ops::transform {

//! Reverse Jordan Wigner transform that transform a Qubit operator to Fermion operator.
// TODO(xusheng): Why cannot use template, otherwise undefined symbol error occur.
// template <typename fermion_t, typename qubit_t>
MQ_NODISCARD fermion_t reverse_jordan_wigner(const qubit_t& ops, int n_qubits = -1);
}  // namespace mindquantum::ops::transform

#endif
