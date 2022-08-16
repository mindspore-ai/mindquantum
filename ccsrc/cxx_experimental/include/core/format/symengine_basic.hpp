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
#include <symengine/serialize-cereal.h>

#include <cereal/archives/json.hpp>
#include <fmt/format.h>

//! Custom formatter for a SymEngine::RCP<const Symengine::Basic>
template <typename char_type>
struct fmt::formatter<SymEngine::RCP<const SymEngine::Basic>, char_type> {
    using basic_t = SymEngine::RCP<const SymEngine::Basic>;

    auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const basic_t& symengine_expr, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", dumps(symengine_expr));
    }

    template <typename T>
    static std::string dumps(SymEngine::RCP<const T> obj) {
        std::ostringstream oss;
        cereal::JSONOutputArchive{oss}(obj);  // NOLINT(whitespace/braces)
        return oss.str();
    }
};

#endif /* FORMAT_SYMENGINE_BASIC_HPP */
