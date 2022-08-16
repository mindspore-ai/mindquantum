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

#ifndef DETAILS_QUBIT_OPERATOR_TERM_POLICY_HPP
#define DETAILS_QUBIT_OPERATOR_TERM_POLICY_HPP

#include <string>
#include <string_view>

#include <symengine/basic.h>

#include <fmt/format.h>

#include "ops/gates/terms_operator.hpp"
#include "pr/parameter_resolver.h"

namespace mindquantum::ops::details {
struct QubitOperatorTermPolicy {
    static auto to_string(const TermValue& value) {
        using namespace std::literals::string_literals;
        if (value == TermValue::X) {
            return "X"s;
        }
        if (value == TermValue::Y) {
            return "Y"s;
        }
        if (value == TermValue::Z) {
            return "Z"s;
        }
        return "UNKNOWN"s;
    }
    static auto to_string(const term_t& term) {
        return fmt::format("{}{}", to_string(std::get<1>(term)), std::get<0>(term));
    }

    static auto parse_terms_string(std::string_view) -> terms_t;

    // TODO(dnguyen): Would need the same based on coefficient type! (ie. SymEngine, ParameterResovler)
    static auto parse_json_complex_double(std::string_view string_data) -> term_dict_t<std::complex<double>>;
};
}  // namespace mindquantum::ops::details

#endif /* DETAILS_QUBIT_OPERATOR_TERM_POLICY_HPP */
