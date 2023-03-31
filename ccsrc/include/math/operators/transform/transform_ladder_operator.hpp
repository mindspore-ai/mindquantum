//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef TRANSFORM_LADDER_OPERATOR_HPP_
#define TRANSFORM_LADDER_OPERATOR_HPP_
#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/qubit_operator_view.hpp"

namespace operators::transform {
using fermion_op_t = fermion::FermionOperator;
using qubit_op_t = qubit::QubitOperator;
using qlist_t = std::vector<size_t>;

qubit_op_t transform_ladder_operator(const fermion::TermValue& value, const qlist_t& x1, const qlist_t& y1,
                                     const qlist_t& z1, const qlist_t& x2, const qlist_t& y2, const qlist_t& z2);
}  // namespace operators::transform
#endif /* TRANSFORM_LADDER_OPERATOR_HPP_ */
