/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef MATH_LONGBITS_LONGBITS_H_
#define MATH_LONGBITS_LONGBITS_H_
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
namespace mindquantum {
class LongBits {
    using ele_t = uint64_t;

 public:
    LongBits() = default;
    explicit LongBits(size_t n_bits);

    // -----------------------------------------------------------------------------

    void operator^=(const LongBits& other);
    void operator&=(const LongBits& other);
    LongBits operator^(const LongBits& other) const;
    LongBits operator&(const LongBits& other) const;
    bool operator==(const LongBits& other) const;

    // -----------------------------------------------------------------------------

    void SetBit(size_t poi, bool val);

    size_t GetBit(size_t poi) const;

    std::string ToString() const;

    void InplaceFlip();
    LongBits Flip();

    bool Any(size_t start, size_t end);

    bool Any(size_t start);

 private:
    LongBits(size_t n_bits, const std::vector<ele_t>& data);

 private:
    size_t n_bits = 1;
    std::vector<ele_t> data = {0};
};
}  // namespace mindquantum

// -----------------------------------------------------------------------------

template <>
struct fmt::formatter<mindquantum::LongBits> {
    constexpr auto parse(format_parse_context& ctx) {  // NOLINT(runtime/references)
        return ctx.begin();
    }
    template <typename FormatContext>
    auto format(const mindquantum::LongBits& obj, FormatContext& ctx) {  // NOLINT(runtime/references)
        return format_to(ctx.out(), obj.ToString());
    }
};
#endif
