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

#ifndef CORE_CONCEPTS_HPP
#define CORE_CONCEPTS_HPP

#include <tuple>

#include "config/concepts.hpp"

#include "experimental/core/config.hpp"
#include "experimental/core/traits.hpp"
#include "experimental/core/types.hpp"

namespace mindquantum {
class CircuitBlock;
}  // namespace mindquantum

namespace mindquantum::concepts {
template <typename T>
concept CircuitLike = (concepts::same_decay_as<circuit_t, T> || concepts::same_decay_as<CircuitBlock, T>);
}  // namespace mindquantum::concepts
#endif /* CORE_CONCEPTS_HPP */
