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

#ifndef FORMAT_STD_OPTIONAL_HPP
#define FORMAT_STD_OPTIONAL_HPP

#include <optional>

#include <fmt/core.h>
#include <fmt/format.h>

template <typename T>
struct fmt::formatter<std::optional<T>> : fmt::formatter<T> {
    template <typename format_context_t>
    auto format(const std::optional<T>& opt, format_context_t& ctx) {
        if (opt) {
            return fmt::formatter<T>::format(*opt, ctx);
        }
        return fmt::format_to(ctx.out(), "<NULL OPTIONAL>");
    }
};

#endif /* FORMAT_STD_OPTIONAL_HPP */
