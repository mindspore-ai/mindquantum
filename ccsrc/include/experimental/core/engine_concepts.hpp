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

#ifndef ENGINE_CONCEPTS_HPP
#define ENGINE_CONCEPTS_HPP

#include <utility>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Target/Device.h>
#include <tweedledum/Target/Mapping.h>

#include "experimental/core/concepts.hpp"

namespace mindquantum::concepts {
template <typename mapper_t>
concept Mapper = requires(mapper_t mapper, tweedledum::Device device, tweedledum::Circuit circuit,
                          tweedledum::Placement placement) {
    { mapper.device() } -> same_decay_as<tweedledum::Device>;
    { mapper.cold_start(device, circuit) } -> std::same_as<std::pair<tweedledum::Circuit, tweedledum::Mapping>>;
    // clang-format off
    { mapper.hot_start(device, circuit, placement) } -> std::same_as<std::pair<tweedledum::Circuit,
                                                                               tweedledum::Mapping>>;
    // clang-format on
};  // NOLINT(readability/braces)
}  // namespace mindquantum::concepts

#endif /* ENGINE_CONCEPTS_HPP */
