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
#include <type_traits>
#include <utility>
#include <vector>

#include <symengine/basic.h>
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
    std::string describe() const override {
        return fmt::format("Equals: {}", comparator_);
    }

    const std::vector<SymEngine::RCP<const SymEngine::Basic>> comparator_;
};

// -----------------------------------------------------------------------------

namespace mq_traits {
template <typename type_t>
struct is_symengine_rcp : std::false_type {};

template <typename type_t>
struct is_symengine_rcp<SymEngine::RCP<type_t>> : std::true_type {};

// clang-format off
template <typename... types_t>
struct are_symengine
    : std::integral_constant<
          bool, ((std::is_base_of_v<SymEngine::Basic, std::remove_cvref_t<types_t>>
                  || is_symengine_rcp<types_t>::value
                  || std::is_same_v<SymEngine::Expression, std::remove_cvref_t<types_t>>)
                 &&...)> {};
// clang-format on

template <typename... types_t>
inline constexpr auto are_symengine_v = are_symengine<types_t...>::value;
}  // namespace mq_traits

template <typename... types_t, typename = std::enable_if_t<mq_traits::are_symengine_v<types_t...>>>
auto Equals(types_t&&... values) {
    return SymEngineArrayMatcher(std::forward<types_t>(values)...);
}
// =============================================================================
}  // namespace mindquantum::catch2

#endif /* MQ_CATCH2_SYMENGINE_HPP */
