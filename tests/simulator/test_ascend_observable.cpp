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

#include <array>
#include <cstdint>
#include <stdexcept>

#include "simulator/vector/ascend/observable.h"

#include <catch2/catch_test_macros.hpp>

namespace mqa = mindquantum::sim::ascend;

TEST_CASE("Ascend observable plans scale with the supplied device core count") {
    struct Case {
        uint32_t cores, shards, leaves;
    };
    // A 20-qubit state has 1024 sampling tiles. Retaining bits 5 and 17
    // gives 16 density entries, each tracing 256 packets of 1024 amplitudes.
    const std::array<Case, 4> cases{{{1, 1, 1024}, {48, 3, 32}, {49, 4, 32}, {128, 8, 8}}};
    for (const auto& test : cases) {
        INFO("Vector cores: " << test.cores);
        const auto density = mqa::CompileDensity(20, (uint64_t{1} << 5) | (uint64_t{1} << 17), test.cores);
        CHECK(density.layout.output_dimension == 4);
        CHECK(density.layout.tile_count == 256);
        CHECK(density.layout.shards == test.shards);

        const auto sampling = mqa::MakeSamplingLayout(20, test.cores);
        CHECK(sampling.bins == test.cores);
        CHECK(sampling.tiles_per_bin == test.leaves);
    }
}

TEST_CASE("Ascend observable plans still reject invalid shapes and empty parallelism") {
    CHECK_THROWS_AS(mqa::CompileDensity(20, 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(mqa::CompileDensity(20, uint64_t{1} << 20, 128), std::invalid_argument);
    CHECK_THROWS_AS(mqa::MakeSamplingLayout(20, 0), std::invalid_argument);
}
