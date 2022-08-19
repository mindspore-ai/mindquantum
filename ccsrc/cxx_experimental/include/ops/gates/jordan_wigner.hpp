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

#ifndef JORDAN_WIGNER_TRANSFORM_H_
#define JORDAN_WIGNER_TRANSFORM_H_

#include "ops/gates/fermion_operator_parameter_resolver.hpp"
#include "ops/gates/qubit_operator_parameter_resolver.hpp"

namespace mindquantum::ops::transform {

MQ_NODISCARD QubitOperatorPR jordan_wigner(const FermionOperatorPR& ops);
}

#endif
