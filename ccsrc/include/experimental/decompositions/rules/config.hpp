//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef DECOPOSITION_RULES_CONFIG_HPP
#define DECOPOSITION_RULES_CONFIG_HPP

#if __has_include(<numbers>) && __cplusplus > 201703L
#    include <numbers>
#endif  // __has_include(<numbers>) && C++20+
#include <string_view>
#include <tuple>

#include "experimental/decompositions/atom_meta.hpp"
#include "experimental/decompositions/config.hpp"

namespace mindquantum::decompositions::rules {
using namespace std::literals::string_view_literals;  // NOLINT(build/namespaces_literals)

#if __has_include(<numbers>) && __cplusplus > 201703L
static constexpr auto PI_VAL = std::numbers::pi;
#else
static constexpr auto PI_VAL = 3.141592653589793;
#endif  // __has_include(<numbers>) && C++20
static constexpr auto PI_VAL_2 = PI_VAL / 2.;
static constexpr auto PI_VAL_4 = PI_VAL / 4.;
}  // namespace mindquantum::decompositions::rules

#endif /* DECOPOSITION_RULES_CONFIG_HPP */
