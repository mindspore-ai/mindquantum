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

#ifndef DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP
#define DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP

#include <string>
#include <string_view>
#include <utility>

#include <symengine/basic.h>

#include <fmt/format.h>

#include "ops/gates/terms_operator.hpp"
#include "pr/parameter_resolver.h"

namespace mindquantum::ops::details {
struct FermionOperatorTermPolicy {
    static auto to_string(const TermValue& value) {
        using namespace std::literals::string_literals;
        if (value == TermValue::adg) {
            return "^"s;
        }
        return ""s;
    }
    static auto hermitian(const terms_t& terms) {
        terms_t new_terms(terms.size());
        size_t n = 0;
        for (auto& [qid, value] : terms) {
            if (value == TermValue::adg) {
                new_terms[terms.size() - n - 1] = std::make_pair(qid, TermValue::a);
            } else {
                new_terms[terms.size() - n - 1] = std::make_pair(qid, TermValue::adg);
            }
            n += 1;
        }
        return new_terms;
    }
    static auto to_string(const term_t& term) {
        return fmt::format("{}{}", std::get<0>(term), to_string(std::get<1>(term)));
    }

    static auto parse_terms_string(std::string_view) -> terms_t;

    // TODO(dnguyen): Would need the same based on coefficient type! (ie. SymEngine, ParameterResovler)
    static auto parse_json_complex_double(std::string_view string_data) -> term_dict_t<std::complex<double>>;
};
}  // namespace mindquantum::ops::details

#endif /* DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP */
