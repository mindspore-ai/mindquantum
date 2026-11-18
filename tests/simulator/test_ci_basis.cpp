/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include <cstdint>
#include <limits>

#include "core/utils.h"
#include "simulator/chemistry/detail/ci_basis.h"

#include <catch2/catch_test_macros.hpp>

using mindquantum::sim::chem::detail::ci_basis::SlaterDeterminant;

TEST_CASE("CI determinants count all 64 occupation bits") {
    CHECK(SlaterDeterminant::count_set_bits(0) == 0);
    CHECK(SlaterDeterminant::count_set_bits(std::numeric_limits<uint64_t>::max()) == 64);
    CHECK(SlaterDeterminant::count_set_bits(uint64_t{0xAAAAAAAA55555555}) == 32);
    for (unsigned bit = 0; bit < 64; ++bit) {
        CHECK(SlaterDeterminant::count_set_bits(uint64_t{1} << bit) == 1);
    }
}

TEST_CASE("Population count overloads preserve signed integer widths") {
    CHECK(mindquantum::CountOne(uint32_t{0}) == 0);
    CHECK(mindquantum::CountOne(uint64_t{0}) == 0);
    CHECK(mindquantum::CountOne(int32_t{0}) == 0);
    CHECK(mindquantum::CountOne(int64_t{0}) == 0);
    CHECK(mindquantum::CountOne(int32_t{-1}) == 32);
    CHECK(mindquantum::CountOne(int64_t{-1}) == 64);
    CHECK(mindquantum::CountOne(std::numeric_limits<int32_t>::min()) == 1);
    CHECK(mindquantum::CountOne(std::numeric_limits<int64_t>::min()) == 1);
    CHECK(mindquantum::CountOne(std::numeric_limits<uint32_t>::max()) == 32);
    CHECK(mindquantum::CountOne(std::numeric_limits<uint64_t>::max()) == 64);
}
