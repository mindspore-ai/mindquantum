/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDQUANTUM_ASCEND_QUBIT_LAYOUT_H_
#define MINDQUANTUM_ASCEND_QUBIT_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "simulator/circuit_tiling/state_space.h"

namespace mindquantum::sim::ascend {
// Compile-time only: amplitudes remain in-place and no layout survives the
// executable boundary. Both predicates and compact bases use this same map.
class QubitLayout {
 public:
    explicit QubitLayout(qbit_t qubits) : positions_(qubits) {
        std::iota(positions_.begin(), positions_.end(), 0);
    }

    uint64_t Physical(uint64_t logical) const {
        uint64_t result = 0;
        for (size_t qubit = 0; qubit < positions_.size(); ++qubit) {
            if (logical & (uint64_t{1} << qubit)) {
                result |= uint64_t{1} << positions_[qubit];
            }
        }
        return result;
    }

    uint64_t Logical(uint64_t physical) const {
        uint64_t result = 0;
        for (size_t qubit = 0; qubit < positions_.size(); ++qubit) {
            if (physical & (uint64_t{1} << positions_[qubit])) {
                result |= uint64_t{1} << qubit;
            }
        }
        return result;
    }

    void AdoptPacked(uint64_t packet, uint64_t logical_targets) {
        // Column bits are physical-order spectators; row bits retain the
        // Builder's logical target order. Both planes use the same bijection.
        auto source = circuit_tiling::QubitsFromMask(packet & ~Physical(logical_targets));
        for (auto qubit : circuit_tiling::QubitsFromMask(logical_targets)) {
            source.push_back(positions_[qubit]);
        }
        const auto destination = circuit_tiling::QubitsFromMask(packet);
        auto permutation = positions_;
        std::iota(permutation.begin(), permutation.end(), 0);
        for (size_t i = 0; i < source.size(); ++i) {
            permutation[source[i]] = destination[i];
        }
        for (auto& position : positions_) {
            position = permutation[position];
        }
    }

    std::vector<std::pair<qbit_t, qbit_t>> Restoration() const {
        auto positions = positions_;
        std::vector<std::pair<qbit_t, qbit_t>> swaps;
        for (qbit_t qubit = 0; qubit < static_cast<qbit_t>(positions.size()); ++qubit) {
            const auto current = positions[qubit];
            if (current == qubit) {
                continue;
            }
            swaps.emplace_back(current, qubit);
            for (auto& position : positions) {
                if (position == current) {
                    position = qubit;
                } else if (position == qubit) {
                    position = current;
                }
            }
        }
        return swaps;
    }

 private:
    qbits_t positions_;
};
}  // namespace mindquantum::sim::ascend
#endif  // MINDQUANTUM_ASCEND_QUBIT_LAYOUT_H_
