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

#ifndef REGISTER_GATE_TYPE_TPP
#define REGISTER_GATE_TYPE_TPP

// clang-format off
#ifndef REGISTER_GATE_TYPE_HPP
#     error This file must only be included by register_gate_type.hpp!
#endif  // REGISTER_GATE_TYPE_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by register_gate_type.hpp
#include "experimental/ops/parametric/register_gate_type.hpp"
// clang-format on

#include <cassert>
#include <vector>

#include "experimental/core/gate_traits.hpp"

// =============================================================================

namespace mindquantum::ops::parametric {
using double_func_t = double (*)(const operator_t&);
using vec_double_func_t = std::vector<double> (*)(const operator_t&);
using params_func_t = param_list_t (*)(const operator_t&);

namespace details {
void register_gate(std::string_view kind, double_func_t angle_func);
void register_gate(std::string_view kind, vec_double_func_t angle_func);
void register_gate(std::string_view kind, params_func_t params_func);
}  // namespace details

template <typename op_t>
void register_gate_type()
#if MQ_HAS_CONCEPTS
    requires((concepts::ParametricGate<op_t>) || (concepts::AngleGate<op_t>) || (concepts::SingleDoubleGate<op_t>)
             || (concepts::MultiDoubleGate<op_t>) )
#endif  // MQ_HAS_CONCEPTS
{
    details::register_gate(op_t::kind(), traits::gate_traits<op_t>::param);
}
}  // namespace mindquantum::ops::parametric

// =============================================================================

#endif /* REGISTER_GATE_TYPE_TPP */
