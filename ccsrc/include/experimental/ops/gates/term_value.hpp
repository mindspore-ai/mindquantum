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

#ifndef TERM_VALUE_HPP
#define TERM_VALUE_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace mindquantum::ops {
enum class TermValue : uint8_t {
    I = 10,
    X = 11,
    Y = 12,
    Z = 13,
    a = 0,
    adg = 1,
};

using term_t = std::pair<uint32_t, TermValue>;
using terms_t = std::vector<term_t>;
using py_term_t = std::pair<uint32_t, uint32_t>;
using py_terms_t = std::vector<py_term_t>;

// -----------------------------------------------------------------------------

// NOLINTNEXTLINE(*avoid-c-arrays,readability-identifier-length)
NLOHMANN_JSON_SERIALIZE_ENUM(TermValue, {
                                            {TermValue::I, "I"},
                                            {TermValue::X, "X"},
                                            {TermValue::Y, "Y"},
                                            {TermValue::Z, "Z"},
                                            {TermValue::a, "v"},
                                            {TermValue::adg, "^"},
                                        });
}  // namespace mindquantum::ops
#endif /* TERM_VALUE_HPP */
