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
#include "math/longbits/longbits.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <fmt/core.h>

namespace mindquantum {
#define REG_OPERATOR(op_eq, op)                                                                                        \
    void LongBits::operator op_eq(const LongBits& other) {                                                             \
        if (other.n_bits != this->n_bits) {                                                                            \
            throw std::runtime_error(                                                                                  \
                fmt::format("n_bits of this ({}) is not equal with other ({})", this->n_bits, other.n_bits));          \
        }                                                                                                              \
        for (size_t i = 0; i < data.size(); i++) {                                                                     \
            data[i] op_eq other.data[i];                                                                               \
        }                                                                                                              \
    }                                                                                                                  \
    LongBits LongBits::operator op(const LongBits& other) const {                                                      \
        auto out = *this;                                                                                              \
        out op_eq other;                                                                                               \
        return out;                                                                                                    \
    }

REG_OPERATOR(^=, ^);
REG_OPERATOR(&=, &);

// -----------------------------------------------------------------------------

LongBits::LongBits(size_t n_bits) : n_bits(n_bits) {
    if (n_bits == 0) {
        throw std::runtime_error("n_bits cannot be zero.");
    }
    constexpr static auto ele_size = sizeof(ele_t) * 8;
    auto n_ele = n_bits / ele_size + ((n_bits % ele_size) != 0);
    data = std::vector<ele_t>(n_ele, 0);
}

LongBits::LongBits(size_t n_bits, const std::vector<ele_t>& data) : n_bits(n_bits), data(data) {
}

// -----------------------------------------------------------------------------

bool LongBits::operator==(const LongBits& other) const {
    return n_bits == other.n_bits && data == other.data;
}

void LongBits::SetBit(size_t poi, bool val) {
    if (poi > n_bits - 1) {
        throw std::runtime_error(fmt::format("poi ({}) out of range: [{}, {}).", poi, 0, n_bits));
    }
    constexpr static auto ele_size = sizeof(ele_t) * 8;
    size_t index_in = poi % ele_size;
    size_t mask = static_cast<uint64_t>(1) << index_in;
    size_t mask_val = static_cast<uint64_t>(val) << index_in;
    ele_t& ele = data[poi / ele_size];
    ele = (ele & ~mask) | mask_val;
}

size_t LongBits::GetBit(size_t poi) const {
    if (poi > n_bits - 1) {
        throw std::runtime_error(fmt::format("poi ({}) out of range: [{}, {}).", poi, 0, n_bits));
    }
    constexpr static auto ele_size = sizeof(ele_t) * 8;
    return (data[poi / ele_size] >> (poi % ele_size)) & 1;
}

std::string LongBits::ToString() const {
    std::string out = "";
    for (size_t i = 0; i < n_bits; i++) {
        out += (GetBit(i) == 0 ? "0" : "1");
    }
    std::reverse(out.begin(), out.end());
    return out;
}

void LongBits::InplaceFlip() {
    std::transform(data.begin(), data.end(), data.begin(), [](const auto& ele) { return ~ele; });
}

LongBits LongBits::Flip() {
    auto out = *this;
    out.InplaceFlip();
    return out;
}

bool LongBits::Any(size_t start, size_t end) {
    if (end <= start) {
        throw std::runtime_error(fmt::format("end ({}) can not be less than start ({}).", end, start));
    }
    if (start >= n_bits) {
        throw std::runtime_error(fmt::format("start ({}) out of range: [{}, {})", start, 0, n_bits));
    }
    end = std::min(end, n_bits);
    for (size_t i = start; i < end; i++) {
        if (GetBit(i) == 1) {
            return true;
        }
    }
    return false;
}

bool LongBits::Any(size_t start) {
    return Any(start, n_bits);
}
#undef REG_OPERATOR
}  // namespace mindquantum
