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

#ifndef FORMAT_PARAMETER_RESOLVER_HPP
#define FORMAT_PARAMETER_RESOLVER_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "core/parameter_resolver.hpp"

//! Custom formatter for a ParameterResolver
template <typename float_t, typename char_type>
struct fmt::formatter<mindquantum::ParameterResolver<float_t>, char_type> {
    using param_resolver_t = mindquantum::ParameterResolver<float_t>;

    auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const param_resolver_t& parameter_resolver, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", parameter_resolver.ToString());
    }
};

#endif /* FORMAT_PARAMETER_RESOLVER_HPP */
