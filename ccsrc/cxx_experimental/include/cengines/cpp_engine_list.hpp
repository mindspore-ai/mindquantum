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

#ifndef CPP_ENGINE_LIST_HPP
#define CPP_ENGINE_LIST_HPP

#include <variant>

#include "cengines/cpp_graph_mapper.hpp"
#include "cengines/cpp_mapping.hpp"
#include "cengines/cpp_optimisation.hpp"
#include "cengines/cpp_printer.hpp"
#include "cengines/cpp_resource_counter.hpp"

namespace mindquantum::cengines {
namespace cpp {
class InstructionFilter {};
class TagRemover {};
}  // namespace cpp

using engine_t = std::variant<cpp::LocalOptimizer, cpp::LinearMapper, cpp::GridMapper, CppGraphMapper, CppPrinter,
                              ResourceCounter, cpp::TagRemover, cpp::InstructionFilter>;
}  // namespace mindquantum::cengines

#endif /* CPP_ENGINE_LIST_HPP */
