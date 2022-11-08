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

#ifndef TRANSFORM_TRANSFORM_LADDER_OPERATOR_HPP
#define TRANSFORM_TRANSFORM_LADDER_OPERATOR_HPP

#include "config/config.hpp"

#include "ops/transform/types.hpp"

namespace mindquantum::ops::transform {
//! Ladder operator transform.
template <typename fermion_op_t>
MQ_NODISCARD to_qubit_operator_t<traits::to_cmplx_type_t<fermion_op_t>> transform_ladder_operator(
    const TermValue& value, const qlist_t& x1, const qlist_t& y1, const qlist_t& z1, const qlist_t& x2,
    const qlist_t& y2, const qlist_t& z2);
}  // namespace mindquantum::ops::transform

#include "transform_ladder_operator.tpp"  // NOLINT

#endif
