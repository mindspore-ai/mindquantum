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

#ifndef FORMAT_SYMENGINE_BASIC_HPP
#define FORMAT_SYMENGINE_BASIC_HPP

#include <string>

#include <symengine/basic.h>
#include <symengine/expression.h>
#include <symengine/serialize-cereal.h>

#include <cereal/archives/json.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace mindquantum::format::details {
template <typename T>
std::string dumps(SymEngine::RCP<const T> obj) {
    std::ostringstream oss;
    cereal::JSONOutputArchive{oss}(obj);  // NOLINT(whitespace/braces)
    return oss.str();
}
template <typename T>
auto loads(const std::string& string_data) {
    SymEngine::RCP<const T> symengine_expr;
    std::istringstream iss{string_data};
    cereal::JSONInputArchive{iss}(symengine_expr);  // NOLINT(whitespace/braces)
    return symengine_expr;
}
}  // namespace mindquantum::format::details

//! Custom formatter for a SymEngine::RCP<const Symengine::Basic>
template <typename char_type>
struct fmt::formatter<SymEngine::RCP<const SymEngine::Basic>, char_type> {
    using basic_t = SymEngine::RCP<const SymEngine::Basic>;

    bool json_output = false;

    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        if (*it == 'j') {
            json_output = true;
            ++it;
        }
        const auto end = ctx.end();

        if (it != end && *it != '}') {
            ctx.error_handler().on_error("invalid type specifier");
        }

        return it;
    }

    template <typename format_context_t>
    auto format(const basic_t& symengine_expr, format_context_t& ctx) const -> decltype(ctx.out()) {
        if (json_output) {
            return fmt::format_to(ctx.out(), "{}", mindquantum::format::details::dumps(symengine_expr));
        }
        return fmt::format_to(ctx.out(), "{}", SymEngine::str(*symengine_expr));
    }
};

//! Custom formatter for a SymEngine::Expression
template <typename char_type>
struct fmt::formatter<SymEngine::Expression, char_type> {
    using expr_t = SymEngine::Expression;

    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const expr_t& symengine_expr, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", SymEngine::str(symengine_expr));
    }
};

// =============================================================================

//! Custom JSON serialization for SymEngine::RCP<const Symengine::Basic>
template <>
struct nlohmann::adl_serializer<SymEngine::RCP<const SymEngine::Basic>> {
    static void to_json(json& json_data, const SymEngine::RCP<const SymEngine::Basic>& symengine_expr) {
        json_data = fmt::format("{:j}", symengine_expr);
        // TODO(dnguyen): This would probably better, but I don't know how to best de-serializing below
        // json_data = nlohmann::json::parse(fmt::format("{}", symengine_expr));
    }

    static void from_json(const json& json_data, SymEngine::RCP<const SymEngine::Basic>& symengine_expr) {
        symengine_expr = mindquantum::format::details::loads<SymEngine::Basic>(json_data.get<std::string>());
    }
};

#endif /* FORMAT_SYMENGINE_BASIC_HPP */
