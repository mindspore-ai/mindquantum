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

#ifndef EXTERNAL_VIEW_HPP
#define EXTERNAL_VIEW_HPP

#include <utility>
#include <vector>

#include "core/config.hpp"

#include "core/circuit_block.hpp"
#include "core/circuit_manager.hpp"

namespace mindquantum::details {
namespace impl {
template <bool reverse>
struct ForeachImpl {
    using manager_t = CircuitManager;
    using block_t = CircuitBlock;

    template <typename Fn>
    static constexpr auto run(const manager_t& manager, Fn&& fn) {
        return manager.foreach_block(std::forward<Fn>(fn));
    }
    template <typename Fn>
    static constexpr auto run(const block_t& block, Fn&& fn) {
        return block.foreach_instruction(std::forward<Fn>(fn));
    }
};

template <>
struct ForeachImpl<true> {
    using manager_t = CircuitManager;
    using block_t = CircuitBlock;

    template <typename Fn>
    static constexpr auto run(const manager_t& manager, Fn&& fn) {
        return manager.foreach_r_block(std::forward<Fn>(fn));
    }
    template <typename Fn>
    static constexpr auto run(const block_t& block, Fn&& fn) {
        return block.foreach_r_instruction(std::forward<Fn>(fn));
    }
};
}  // namespace impl

// ------------------------------------------------------------------------

class ExternalView {
 public:
    using manager_t = CircuitManager;
    using block_t = CircuitBlock;
    using cbit_t = tweedledum::Cbit;
    using qubit_t = tweedledum::Qubit;
    using instruction_t = tweedledum::Instruction;

    explicit ExternalView(const manager_t& manager) : manager_{manager} {
    }

    template <typename Fn>
    void foreach_instruction(Fn&& fn) const {
        foreach_<false>(std::forward<Fn>(fn));
    }

    template <typename Fn>
    void foreach_r_instruction(Fn&& fn) const {
        foreach_<true>(std::forward<Fn>(fn));
    }

 private:
    template <bool reverse, typename Fn>
    void foreach_(Fn&& fn) const {
        static_assert(std::is_invocable_r_v<void, Fn, const instruction_t&>);

        impl::ForeachImpl<reverse>::run(manager_, [fn](const block_t& block) {
            std::vector<qubit_t> qubits;
            std::vector<cbit_t> cbits;
            impl::ForeachImpl<reverse>::run(block, [&fn, &block, &qubits, &cbits](const instruction_t& inst) {
                inst.foreach_qubit(
                    [&qubits, &block](const qubit_t& qubit) { qubits.emplace_back(block.translate_id_td(qubit)); });
                inst.foreach_cbit(
                    [&cbits, &block](const cbit_t& cbit) { cbits.emplace_back(block.translate_id_td(cbit)); });

                fn(instruction_t(inst, qubits, cbits));
                qubits.clear();
                cbits.clear();
            });
        });
    }

    const manager_t& manager_;
};

class ExternalBlockView {
 public:
    using block_t = CircuitBlock;
    using instruction_t = tweedledum::Instruction;
    using cbit_t = tweedledum::Cbit;
    using qubit_t = tweedledum::Qubit;

    explicit ExternalBlockView(const block_t& block) : block_{block} {
    }

    template <typename Fn>
    void foreach_instruction(Fn&& fn) const {
        std::vector<qubit_t> qubits;
        std::vector<cbit_t> cbits;
        block_.foreach_instruction([&fn = fn, &block = block_, &qubits, &cbits](const instruction_t& inst) {
            inst.foreach_qubit(
                [&qubits, &block](const qubit_t& qubit) { qubits.emplace_back(block.translate_id_td(qubit)); });
            inst.foreach_cbit(
                [&cbits, &block](const cbit_t& cbit) { cbits.emplace_back(block.translate_id_td(cbit)); });

            fn(instruction_t(inst, qubits, cbits));
            qubits.clear();
            cbits.clear();
        });
    }

    template <typename Fn>
    void foreach_r_instruction(Fn&& fn) const {
        std::vector<qubit_t> qubits;
        std::vector<cbit_t> cbits;
        block_.foreach_r_instruction([&fn = fn, &block = block_, &qubits, &cbits](const instruction_t& inst) {
            inst.foreach_qubit(
                [&wires = qubits, &block](const qubit_t& qubit) { wires.emplace_back(block.translate_id_td(qubit)); });
            inst.foreach_cbit(
                [&wires = cbits, &block](const cbit_t& cbit) { wires.emplace_back(block.translate_id_td(cbit)); });

            // TODO(dnguyen): Can we get rid of the temporary variable here?
            fn(instruction_t(inst, qubits, cbits));
            qubits.clear();
            cbits.clear();
        });
    }

 private:
    const block_t& block_;
};
}  // namespace mindquantum::details

#endif /* EXTERNAL_VIEW_HPP */
