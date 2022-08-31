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

#include "ops/gates/fermion_operator_parameter_resolver.hpp"
#include "ops/gates/qubit_operator_parameter_resolver.hpp"

namespace mindquantum::ops::transform {
using qubit_t = QubitOperatorPR;
using fermion_t = FermionOperatorPR;
using qlist_t = std::vector<term_t::first_type>;
}  // namespace mindquantum::ops::transform
#endif
