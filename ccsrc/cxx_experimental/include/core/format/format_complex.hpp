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

#ifndef FORMAT_COMPLEX_HPP
#define FORMAT_COMPLEX_HPP

#include <complex>

#include <fmt/format.h>

template <typename T, typename Char>
struct fmt::formatter<std::complex<T>, Char> : public fmt::formatter<T, Char> {
    using base = fmt::formatter<T, Char>;
    fmt::detail::dynamic_format_specs<Char> specs_;
    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
    template <typename FormatCtx>
    auto format(const std::complex<T>& number, FormatCtx& ctx) -> decltype(ctx.out()) {
        const auto& real = number.real();
        const auto& imag = number.imag();
        if (real && !imag) {
            return base::format(real, ctx);
        }
        if (!real && imag) {
            base::format(imag, ctx);
            return format_to(ctx.out(), "j");  // NB: use j instead of i (ie. like Python)
        }

        format_to(ctx.out(), "(");
        base::format(real, ctx);
        if (imag) {
            if (number.real() && number.imag() >= 0 && specs_.sign != sign::plus) {
                format_to(ctx.out(), "+");
            }
            base::format(imag, ctx);
            format_to(ctx.out(), "j");  // NB: use j instead of i (ie. like Python)
        }
        return format_to(ctx.out(), ")");
    }
};

#endif /* FORMAT_COMPLEX_HPP */
