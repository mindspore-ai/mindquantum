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

#ifndef CIRCUIT_BLOCK_TPP
#define CIRCUIT_BLOCK_TPP

#ifndef CIRCUIT_BLOCK_HPP
#    error This file must only be included by circuit_block.hpp!
#endif  // !CIRCUIT_MANAGER_HPP

#include <algorithm>
#include <utility>
#include <vector>

// clang-format off
// NB: This is mainly for syntax checkers and completion helpers as this file is only intended to be included directly
//     by circuit_manager.hpp
#include "experimental/core/circuit_block.hpp"
// clang-format on

#include <tweedledum/Operators/Standard/Swap.h>
#include <tweedledum/Passes/Utility/shallow_duplicate.h>

#include "experimental/mapping/partial_placer.hpp"

// =============================================================================

//! Custom formatter for a QubitID
template <typename char_type>
struct fmt::formatter<mindquantum::QubitID, char_type> {
    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const mindquantum::QubitID& qubit_id, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Q[{}]", static_cast<uint32_t>(qubit_id));
    }
};

// =============================================================================

namespace mindquantum {
template <typename OpT>
auto CircuitBlock::apply_operator(OpT&& optor, const qubit_ids_t& control_qubits, const qubit_ids_t& target_qubits)
    -> inst_ref_t {
    return circuit_.apply_operator(std::forward<OpT>(optor), translate_ext_ids_(control_qubits, target_qubits));
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
#if MQ_HAS_CONCEPTS
template <concepts::cold_start_t Fn, concepts::hot_start_t Gn>
#else
template <typename Fn, typename Gn>
#endif  // MQ_HAS_CONCEPTS
void CircuitBlock::apply_mapping(const device_t& device, const Fn& cold_start, const Gn& hot_start) {
    // clang-format off
          using mapping_ret_t = std::pair<circuit_t, mapping_t>;
          static_assert(std::is_invocable_r_v<mapping_ret_t, Fn, const device_t&, const circuit_t&>);
          static_assert(std::is_invocable_r_v<mapping_ret_t, Gn, const device_t&, const circuit_t&, placement_t&>);
    // clang-format on

    assert(device.num_qubits() > 0);
    if (!device_) {
        device_ = &device;
    } else {
        // TODO(dnguyen): Better error handling
        assert(&device == device_);
    }

    const auto n_qubits_orig = circuit_.num_qubits();

    circuit_t mapped;

    // ---------------------------
    /* Calculate the mapping
     *
     * This results in a circuit with the operations in reverse order since the instructions are stored in a
     * single-linked list with back pointers.
     */
    if (!has_mapping()) {
        std::tie(mapped, mapping_) = cold_start(*device_, circuit_);
    } else {
        auto& mapping = mapping_.value();
        mapping.placement.reset();

        std::vector<qubit_t> new_qubits;
        auto has_new_qubits = false;
        for (auto v(0UL); v < std::size(mapping.placement.v_to_phy()); ++v) {
            const auto phy = mapping.init_placement.v_to_phy(v);
            if (phy != qubit_t::invalid()) {
                mapping.placement.map_v_phy(v, phy);
            } else {
                new_qubits.emplace_back(v);
            }
        }

        if (!std::empty(new_qubits)) {
            mapping::PartialPlacer placer(*device_, mapping.placement);
            placer.run(new_qubits);
        }

        mapping_t new_mapping(mapping.init_placement);
        std::tie(mapped, new_mapping) = hot_start(device, circuit_, mapping.placement);

        assert(mapping.init_placement == new_mapping.init_placement);
        mapping.placement = new_mapping.placement;
    }

    circuit_ = std::move(mapped);

    // ---------------------------
    // Recalculate the new internal mappings since wire IDs in circuit_ will now be physical IDs
    // NB: this ignores "ghost" qubits
    std::vector<qubit_t> old_to_mapped(circuit_.num_wires(), qubit_t::invalid());
    circuit_.foreach_qubit([&old_to_mapped = old_to_mapped, &placement = mapping_->placement](const qubit_t& qubit) {
        old_to_mapped[qubit] = placement.v_to_phy(qubit);
    });

    update_mappings_(old_to_mapped);
}
}  // namespace mindquantum

// =============================================================================

namespace mindquantum {
template <typename Fn>
void CircuitBlock::foreach_instruction(Fn&& fn) const {
    circuit_.foreach_instruction(std::forward<Fn>(fn));
}

template <typename Fn>
void CircuitBlock::foreach_r_instruction(Fn&& fn) const {
    circuit_.foreach_r_instruction(std::forward<Fn>(fn));
}

template <typename Fn>
void CircuitBlock::transform(const Fn& fn) {
    // clang-format off
          static_assert(std::is_invocable_r_v<circuit_t, Fn, const circuit_t&> ||
                        std::is_invocable_r_v<void, Fn, circuit_t&>);
    // clang-format on
    if constexpr (std::is_invocable_r_v<circuit_t, Fn, const circuit_t&>) {
        circuit_ = fn(circuit_);
    } else {
        fn(circuit_);
    }
}
}  // namespace mindquantum

// =============================================================================

#endif /* CIRCUIT_BLOCK_TPP */
