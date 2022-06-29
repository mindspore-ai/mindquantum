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

#ifndef CIRCUIT_MANAGER_TPP
#define CIRCUIT_MANAGER_TPP

#ifndef CIRCUIT_MANAGER_HPP
#    error This file must only be included by circuit_manager.hpp!
#endif  // !CIRCUIT_MANAGER_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by circuit_manager.hpp
#include <algorithm>
#include <numeric>
#include <utility>

#include <tweedledum/Operators/Standard/Swap.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "circuit_manager.hpp"
#include "core/details/detected.hpp"

// =============================================================================

namespace mindquantum {
template <typename OpT>
auto CircuitManager::apply_operator(OpT&& optor, const qubit_ids_t& control_qubits, const qubit_ids_t& target_qubits)
    -> inst_ref_t {
    return blocks_.back().apply_operator(std::forward<OpT>(optor), control_qubits, target_qubits);
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
#if MQ_HAS_CONCEPTS
template <concepts::Mapper mapper_t>
#else
template <typename mapper_t>
#endif  // MQ_HAS_CONCEPTS
void CircuitManager::apply_mapping(const mapper_t& mapper) {
    apply_mapping(
        mapper.device(),
        [&mapper](const device_t& device, const circuit_t& circuit) { return mapper.cold_start(device, circuit); },
        [&mapper](const device_t& device, const circuit_t& circuit, placement_t& placement) {
            return mapper.hot_start(device, circuit, placement);
        });
}

#if MQ_HAS_CONCEPTS
template <concepts::cold_start_t Fn, concepts::hot_start_t Gn>
#else
template <typename Fn, typename Gn>
#endif  // MQ_HAS_CONCEPTS
void CircuitManager::apply_mapping(const device_t& device, Fn&& cold_start, Gn&& hot_start) {
    blocks_.back().apply_mapping(device, std::forward<Fn>(cold_start), std::forward<Gn>(hot_start));
}

}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
template <typename Fn>
void CircuitManager::foreach_block(const Fn& fn) const {
    static_assert(std::is_invocable_r_v<void, Fn, const CircuitBlock&>);

    std::for_each(std::cbegin(blocks_), std::cend(blocks_) - 1, fn);
}

template <typename Fn>
void CircuitManager::foreach_r_block(const Fn& fn) const {
    static_assert(std::is_invocable_r_v<void, Fn, const CircuitBlock&>);

    std::for_each(std::crbegin(blocks_) + 1, std::crend(blocks_), fn);
}

template <typename Fn>
void CircuitManager::foreach_instruction(Fn&& fn, uncommitted_t) const {
    blocks_.back().foreach_instruction(std::forward<Fn>(fn));
}

template <typename Fn>
void CircuitManager::transform(Fn&& fn) {
    blocks_.back().transform(std::forward<Fn>(fn));
}
}  // namespace mindquantum

// =============================================================================

#endif /* CIRCUIT_MANAGER_TPP */
