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

#include <algorithm>
#include <limits>
#include <vector>

#include "simulator/state_index.h"

#include <catch2/catch_test_macros.hpp>

namespace state_index = mindquantum::sim::state_index;
using mindquantum::index_t;

TEST_CASE("State-index windows preserve the full expanded index") {
    const state_index::StateIndexWindow window(5, {1, 4, 8});

    CHECK(window.WindowCapacity() == 8);
    CHECK(window.LocalStateMask() == 31);
    CHECK(window.LocalExcludedQubits() == mindquantum::qbits_t{1, 4});
    for (index_t compressed = 0; compressed < 64; ++compressed) {
        const auto compressed_origin = window.CompressedWindowOrigin(compressed);
        const auto state_origin = window.StateWindowOrigin(compressed_origin);
        const auto local_state_index = window.ExpandLocal(window.LocalCompressedIndex(compressed));
        CHECK(window.Expand(compressed) == state_origin + local_state_index);
        CHECK((state_origin & window.LocalStateMask()) == 0);
        CHECK(local_state_index <= window.LocalStateMask());
    }
}

TEST_CASE("State-index traversal covers non-divisible windows exactly once") {
    const state_index::StateIndexWindow window(5, {1, 4, 8});
    const state_index::StateIndexBatchTraversal traversal(window, 25, 6);
    std::vector<unsigned> visits(25, 0);

    CHECK(traversal.BatchCount() == 7);
    for (index_t ordinal = 0; ordinal < traversal.BatchCount(); ++ordinal) {
        const auto batch = traversal.Batch(ordinal);
        const auto active_end = std::min(batch.local_compressed_start + 6, batch.local_compressed_count);
        for (index_t local = batch.local_compressed_start; local < active_end; ++local) {
            ++visits[batch.compressed_window_origin + local];
        }
    }
    for (auto visit_count : visits) {
        CHECK(visit_count == 1);
    }
}

TEST_CASE("State-index traversal keeps the uniform fast path aligned") {
    const state_index::StateIndexWindow window(5, {1, 4, 8});
    const state_index::StateIndexBatchTraversal traversal(window, 23, 4);

    REQUIRE(traversal.BatchCount() == 6);
    CHECK(traversal.Batch(0).compressed_window_origin == 0);
    CHECK(traversal.Batch(1).local_compressed_start == 4);
    CHECK(traversal.Batch(2).compressed_window_origin == 8);
    CHECK(traversal.Batch(4).compressed_window_origin == 16);
    CHECK(traversal.Batch(5).local_compressed_count == 7);
}

TEST_CASE("FP32 state-index windows continue beyond the gather encoding boundary") {
    const state_index::StateIndexWindow window(31, {0});
    const auto window_capacity = index_t{1} << 30;
    const state_index::StateIndexBatchTraversal traversal(window, window_capacity + 17, 8);

    CHECK(window.WindowCapacity() == window_capacity);
    const auto last_first_window = traversal.Batch(window_capacity / 8 - 1);
    CHECK(last_first_window.compressed_window_origin == 0);
    CHECK(last_first_window.local_compressed_start == window_capacity - 8);

    const auto first_second_window = traversal.Batch(window_capacity / 8);
    CHECK(first_second_window.compressed_window_origin == window_capacity);
    CHECK(first_second_window.local_compressed_start == 0);
    CHECK(first_second_window.local_compressed_count == 17);
    CHECK(window.StateWindowOrigin(window_capacity) == (index_t{1} << 31));
}

TEST_CASE("State-index metadata fails fast on invalid layouts") {
    const auto index_bits = std::numeric_limits<index_t>::digits;
    CHECK_THROWS_AS(state_index::StateIndexWindow(0, {}), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexWindow(index_bits, {}), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexWindow(5, {2, 1}), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexWindow(5, {2, 2}), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexWindow(5, {-1}), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexWindow(5, {static_cast<mindquantum::qbit_t>(index_bits)}),
                    std::invalid_argument);

    const state_index::StateIndexWindow window(5, {});
    CHECK_THROWS_AS(state_index::StateIndexBatchTraversal(window, 0, 4), std::invalid_argument);
    CHECK_THROWS_AS(state_index::StateIndexBatchTraversal(window, 8, 0), std::invalid_argument);
}
