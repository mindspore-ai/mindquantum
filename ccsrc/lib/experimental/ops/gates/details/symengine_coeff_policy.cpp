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

#include "experimental/ops/gates/details/symengine_coeff_policy.hpp"

#include <optional>
#include <sstream>
#include <string>

#include <symengine/basic.h>
#include <symengine/serialize-cereal.h>

#include <cereal/archives/json.hpp>

#include "experimental/core/logging.hpp"

// ==============================================================================

namespace {
template <typename T>
auto loads(const std::string& string_data) {
    MQ_INFO("Attempting to parse: '{}'", string_data);
    SymEngine::RCP<const T> symengine_expr;
    std::istringstream iss{string_data};
    cereal::JSONInputArchive{iss}(symengine_expr);  // NOLINT(whitespace/braces)
    return symengine_expr;
}
}  // Namespace

namespace mindquantum::ops::details {
auto SymEngineCoeffPolicy::coeff_from_string(const boost::iterator_range<std::string_view::const_iterator>& range)
    -> std::optional<coeff_t> {
    const std::string data{std::begin(range), std::end(range)};
    MQ_INFO("Attempting to parse: '{}'", data);
    try {
        return loads<SymEngine::Basic>(data);
    } catch (std::exception& e) {
        MQ_ERROR("Caught exception: {}", e.what());
        return {};
    } catch (...) {
        MQ_ERROR("Caught unexpected exception!");
        return {};
    }
}
}  // namespace mindquantum::ops::details

// ==============================================================================
