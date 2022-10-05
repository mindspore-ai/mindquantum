//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef REGISTER_GATE_TYPE_HPP
#define REGISTER_GATE_TYPE_HPP

#include <vector>

#include <tweedledum/IR/Instruction.h>

#include "experimental/ops/parametric/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/gate_concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::ops::parametric {
//! Register a new gate class
/*!
 * \tparam operator_t Type of the gate to register
 *
 * \note If the gate is neither a parametric gate or a gate with an angle() method, this method is no-op.
 */
template <typename op_t>
void register_gate_type() MQ_REQUIRES((concepts::ParametricGate<op_t>) || (concepts::AngleGate<op_t>)
                                      || (concepts::SingleDoubleGate<op_t>) || (concepts::MultiDoubleGate<op_t>) );

//! Get the parameters of an operation
/*!
 * \param optor A quantum operation
 */
[[nodiscard]] gate_param_t get_param(const operator_t& optor) noexcept;
}  // namespace mindquantum::ops::parametric

#include "register_gate_type.tpp"

#endif /* REGISTER_GATE_TYPE_HPP */
