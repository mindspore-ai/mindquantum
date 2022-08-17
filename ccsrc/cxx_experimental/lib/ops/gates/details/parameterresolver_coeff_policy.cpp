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

#include "ops/gates/details/parameterresolver_coeff_policy.hpp"

#include <optional>

#include <boost/spirit/home/x3.hpp>

#include "core/logging.hpp"
#include "pr/parameter_resolver.h"

// ==============================================================================

namespace x3 = boost::spirit::x3;

namespace mindquantum::ops::details {
auto DoublePRCoeffPolicy::coeff_from_string(const boost::iterator_range<std::string_view::const_iterator> &range)
    -> std::optional<coeff_t> {
    MQ_INFO("Attempting to parse: '{}'", std::string(range.begin(), range.end()));
    return {};
}

// -----------------------------------------------------------------------------

auto CmplxDoublePRCoeffPolicy::coeff_from_string(const boost::iterator_range<std::string_view::const_iterator> &range)
    -> std::optional<coeff_t> {
    MQ_INFO("Attempting to parse: '{}'", std::string(range.begin(), range.end()));
    return {};
}
}  // namespace mindquantum::ops::details

// ==============================================================================
