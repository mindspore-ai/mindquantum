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
#include "math/operators/fermion_operator_view.hpp"
#include "math/tensor/ops.hpp"
namespace operators::fermion {
namespace tn = tensor;
using fermion_op_t = fermion::FermionOperator;

fermion_op_t fermion_number_operator(int n_modes, int mode = -1,
                                     const parameter::ParameterResolver& coeff = tn::ops::ones(1));
}  // namespace operators::fermion
#endif
