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

#ifndef OPS_CONFIG_HPP
#define OPS_CONFIG_HPP

#include <variant>
#include <vector>

#include <symengine/basic.h>
#include <symengine/eval_double.h>
#include <symengine/expression.h>

#include "experimental/core/config.hpp"

namespace mindquantum::ops::parametric {
using subs_map_t = SymEngine::map_basic_basic;
using basic_t = SymEngine::RCP<const SymEngine::Basic>;
using double_list_t = std::vector<double>;
using param_list_t = SymEngine::vec_basic;
using gate_param_t = std::variant<std::monostate, double, double_list_t, param_list_t>;
}  // namespace mindquantum::ops::parametric

#endif /* OPS_CONFIG_HPP */
