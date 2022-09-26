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

#ifndef MQ_PYTHON_CORE_FORMAT_PYTYPES_HPP
#define MQ_PYTHON_CORE_FORMAT_PYTYPES_HPP

#include <cstdint>
#include <string>

#include <fmt/ranges.h>
#include <pybind11/pytypes.h>

//! Very basic fmtlib formatter for Pybind11 Python types
template <typename char_type>
struct fmt::formatter<pybind11::handle, char_type> {
    auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const pybind11::handle& src, format_context_t& ctx) const -> decltype(ctx.out()) {
        if (pybind11::isinstance<pybind11::int_>(src)) {
            return fmt::format_to(ctx.out(), "{}", static_cast<int64_t>(src.cast<pybind11::int_>()));
        }
        if (pybind11::isinstance<pybind11::float_>(src)) {
            return fmt::format_to(ctx.out(), "{}", static_cast<double>(src.cast<pybind11::float_>()));
        }
        if (pybind11::isinstance<pybind11::set>(src)) {
            return fmt::format_to(ctx.out(), "<set>{{{}}}", src.cast<pybind11::set>());
        }
        if (pybind11::isinstance<pybind11::dict>(src)) {
            return fmt::format_to(ctx.out(), "<dict>{}", src.cast<pybind11::dict>());
        }
        if (pybind11::isinstance<pybind11::list>(src)) {
            return fmt::format_to(ctx.out(), "<list>{}", src.cast<pybind11::list>());
        }
        if (pybind11::isinstance<pybind11::tuple>(src)) {
            return fmt::format_to(ctx.out(), "<tuple>({})", src.cast<pybind11::tuple>());
        }
        if (pybind11::isinstance<pybind11::str>(src)) {
            return fmt::format_to(ctx.out(), "'{}'", static_cast<std::string>(src.cast<pybind11::str>()));
        }
        if (pybind11::isinstance<pybind11::none>(src)) {
            return fmt::format_to(ctx.out(), "None");
        }
        return fmt::format_to(ctx.out(), "UNKNOWN");
    }
};

#endif /* MQ_PYTHON_CORE_FORMAT_PYTYPES_HPP */
