/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MQ_FORMAT_COMPLEX_HPP
#define MQ_FORMAT_COMPLEX_HPP

#include <complex>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

//! Custom formatter for a std::complex<T>
template <typename float_t, typename char_type>
struct fmt::formatter<std::complex<float_t>, char_type> : public fmt::formatter<float_t, char_type> {
    using base = fmt::formatter<float_t, char_type>;
    fmt::detail::dynamic_format_specs<char_type> specs_;
    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        using handler_type = fmt::detail::dynamic_specs_handler<format_parse_context>;
        auto type = fmt::detail::type_constant<float_t, char_type>::value;
        fmt::detail::specs_checker<handler_type> handler(handler_type(specs_, ctx), type);
        parse_format_specs(ctx.begin(), ctx.end(), handler);
        return base::parse(ctx);
    }
    template <typename FormatCtx>
    auto format(const std::complex<float_t>& number, FormatCtx& ctx) const -> decltype(ctx.out()) {
        const auto& real = number.real();
        const auto& imag = number.imag();
        if (real && !imag) {
            return base::format(real, ctx);
        }
        if (!real && imag) {
            base::format(imag, ctx);
            return fmt::format_to(ctx.out(), "j");  // NB: use j instead of i (ie. like Python)
        }

        fmt::format_to(ctx.out(), "(");
        base::format(real, ctx);
        if (imag) {
            if (number.real() && number.imag() >= 0 && specs_.sign != sign::plus) {
                fmt::format_to(ctx.out(), "+");
            }
            base::format(imag, ctx);
            fmt::format_to(ctx.out(), "j");  // NB: use j instead of i (ie. like Python)
        }
        return fmt::format_to(ctx.out(), ")");
    }
};

// =============================================================================

//! Custom JSON serialization for std::complex<T>
template <typename T>
struct nlohmann::adl_serializer<std::complex<T>> {
    static void to_json(json& json_data, const std::complex<T>& number) {
        json_data = nlohmann::json{number.real(), number.imag()};
    }

    static void from_json(const json& json_data, std::complex<T>& number) {
        number.real(json_data.at(0).get<T>());
        number.imag(json_data.at(1).get<T>());
    }
};

#endif /* MQ_FORMAT_COMPLEX_HPP */
