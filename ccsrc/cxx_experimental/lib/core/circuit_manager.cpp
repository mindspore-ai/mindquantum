//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include "core/circuit_manager.hpp"

#include <limits>

#include "ops/gates/measure.hpp"

// =============================================================================

namespace mindquantum {
CircuitManager::CircuitManager() : blocks_{{}} {
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
void CircuitManager::commit_changes() {
    blocks_.emplace_back(blocks_.back(), CircuitBlock::chain_ctor);
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
std::size_t CircuitManager::size(committed_t) const {
    return std::accumulate(std::begin(blocks_), std::end(blocks_) - 1, 0UL,
                           [](const auto& init, const auto& block) { return init + std::size(block); });
}

std::size_t CircuitManager::size(uncommitted_t) const {
    return std::size(blocks_.back());
}

std::size_t CircuitManager::size() const {
    return std::accumulate(std::begin(blocks_), std::end(blocks_), 0UL,
                           [](const auto& init, const auto& block) { return init + std::size(block); });
}

bool CircuitManager::has_mapping() const {
    return blocks_.front().has_mapping();
}

bool CircuitManager::has_qubit(ext_id_t qubit_id) const {
    return blocks_.back().has_qubit(qubit_id);
}

qubit_id_t CircuitManager::translate_id(const td_qid_t& ref) const {
    return blocks_.back().translate_id(ref);
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
bool CircuitManager::add_qubit(ext_id_t ext_qubit_id) {
    return blocks_.back().add_qubit(ext_qubit_id);
}

void CircuitManager::delete_qubits(const std::vector<ext_id_t>& ids_to_delete) {
    auto& block = blocks_.back();
    if (std::size(block) == 0) {
        /* The last block is still empty -> replace it!
         *
         * We do this in three steps:
         *   1. Figure out which qubit should remain by building a qubit inclusion list from the current last
         *      block and the exclusion list passed in argument.
         *   2. Remove the last block.
         *   3. Build a qubit exclusion list based on the new last block and the qubit inclusion list we built
         *      during step 1, or if we only have a single block, create a new empty block and add the required
         *      qubits to it.
         */

        // 1
        auto ids = blocks_.back().ext_ids();
        std::vector<ext_id_t> ids_to_keep;
        std::for_each(std::begin(ids), std::end(ids), [&ids_to_keep, &ids_to_delete](const auto& qubit) {
            for (const auto& to_delete : ids_to_delete) {
                if (qubit == to_delete) {
                    return;
                }
            }
            ids_to_keep.emplace_back(qubit);
        });

        // 2
        blocks_.pop_back();

        // 3
        if (std::empty(blocks_)) {
            blocks_.emplace_back();
            std::for_each(std::begin(ids_to_keep), std::end(ids_to_keep),
                          [&block = blocks_.back()](const auto& qubit) { block.add_qubit(qubit); });
        } else {
            ids = blocks_.back().ext_ids();
            std::vector<ext_id_t> exclude_ids;
            std::for_each(std::begin(ids), std::end(ids), [&exclude_ids, &ids_to_keep](const auto& qubit) {
                for (const auto& to_keep : ids_to_keep) {
                    if (qubit == to_keep) {
                        return;
                    }
                }
                exclude_ids.emplace_back(qubit);
            });
            blocks_.emplace_back(blocks_.back(), exclude_ids, CircuitBlock::chain_ctor);
        }
    } else {
        blocks_.emplace_back(blocks_.back(), ids_to_delete, CircuitBlock::chain_ctor);
    }
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
auto CircuitManager::apply_operator(const instruction_t& optor, const qureg_t& control_qubits,
                                    const qureg_t& target_qubits) -> inst_ref_t {
    return blocks_.back().apply_operator(optor, control_qubits, target_qubits);
}

auto CircuitManager::apply_measurement(qubit_id_t id) -> inst_ref_t {
    return blocks_.back().apply_measurement(id);
}
}  // namespace mindquantum

// =============================================================================
namespace mindquantum {
details::ExternalView CircuitManager::as_projectq(committed_t) const {
    return details::ExternalView(*this);
}

details::ExternalBlockView CircuitManager::as_projectq(uncommitted_t) const {
    return details::ExternalBlockView(blocks_.back());
}

details::PhysicalView CircuitManager::as_physical(committed_t) const {
    return details::PhysicalView(*this);
}
}  // namespace mindquantum

// =============================================================================
