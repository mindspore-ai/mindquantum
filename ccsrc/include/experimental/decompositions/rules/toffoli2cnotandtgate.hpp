//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef DECOMPOSITION_RULE_TOFFOLI2CNOTANDTGATE_HPP
#define DECOMPOSITION_RULE_TOFFOLI2CNOTANDTGATE_HPP

#include <tuple>

#include <tweedledum/Operators/Standard/T.h>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"

namespace mindquantum::decompositions::rules {
class Toffoli2CNOTAndT
    : public GateDecompositionRule<Toffoli2CNOTAndT, std::tuple<ops::X>, SINGLE_TGT_DOUBLE_CTRL, atoms::C<ops::X>,
                                   ops::H, ops::T, ops::Tdg> {
 public:
    static_assert(self_t::num_controls_for_decomp == 2);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "Toffoli2CNOTAndT"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        assert(std::size(qubits) == 3);
        const auto& control0 = qubits[0];
        const auto& control1 = qubits[1];
        const auto& target = qubits[2];

        atom<ops::H>()->apply(circuit, ops::H{}, {target});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control0, target});
        atom<ops::T>()->apply(circuit, ops::T{}, {control0});
        atom<ops::Tdg>()->apply(circuit, ops::Tdg{}, {target});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control1, target});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control1, control0});
        atom<ops::Tdg>()->apply(circuit, ops::Tdg{}, {control0});
        atom<ops::T>()->apply(circuit, ops::T{}, {target});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control1, control0});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control0, target});
        atom<ops::Tdg>()->apply(circuit, ops::Tdg{}, {target});
        atom<atoms::C<ops::X>>()->apply(circuit, ops::X{}, {control1, target});
        atom<ops::T>()->apply(circuit, ops::T{}, {target});
        atom<ops::T>()->apply(circuit, ops::T{}, {control1});
        atom<ops::H>()->apply(circuit, ops::X{}, {target});
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_TOFFOLI2CNOTANDTGATE_HPP */
