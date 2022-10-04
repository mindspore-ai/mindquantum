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

#ifndef MQ_CATCH2_SYMENGINE_HPP
#define MQ_CATCH2_SYMENGINE_HPP

#include <string>
#include <utility>
#include <vector>

#include <symengine/expression.h>

#include <fmt/ranges.h>

#include "config/format/symengine.hpp"

#include "mindquantum/catch2/catch2_fmt_formatter.hpp"
#include "mindquantum/catch2/mindquantum.hpp"

#include <catch2/catch_tostring.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

// =============================================================================

namespace Catch {
template <>
struct StringMaker<SymEngine::RCP<const SymEngine::Basic>>
    : mindquantum::details::FmtStringMakerBase<SymEngine::RCP<const SymEngine::Basic>> {};
}  // namespace Catch

// =============================================================================

namespace mindquantum::catch2 {
struct SymEngineArrayMatcher : Catch::Matchers::MatcherGenericBase {
    template <typename... Ts>
    explicit SymEngineArrayMatcher(Ts&&... ts) : comparator_{{std::forward<Ts>(ts)...}} {
    }

    bool match(const std::vector<SymEngine::RCP<const SymEngine::Basic>>& v) const {
        if (comparator_.size() != v.size()) {
            return false;
        }
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (!eq(*comparator_[i], *v[i])) {
                return false;
            }
        }
        return true;
    }
    std::string describe() const {
        return fmt::format("Equals: {}", comparator_);
    }

    const std::vector<SymEngine::RCP<const SymEngine::Basic>> comparator_;
};

// -----------------------------------------------------------------------------

template <typename... Ts>
auto Equals(Ts&&... ts) {
    return SymEngineArrayMatcher(std::forward<Ts>(ts)...);
}
// =============================================================================
}  // namespace mindquantum::catch2

#endif /* MQ_CATCH2_SYMENGINE_HPP */
