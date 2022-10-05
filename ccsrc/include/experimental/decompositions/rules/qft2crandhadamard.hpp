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

#ifndef DECOMPOSITION_RULE_QFT2CRANDHADAMARD_HPP
#define DECOMPOSITION_RULE_QFT2CRANDHADAMARD_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

namespace mindquantum::decompositions::rules {
class QFT2CrAndHadamard
    : public GateDecompositionRule<QFT2CrAndHadamard, std::tuple<ops::X>, ANY_TGT_NO_CTRL, ops::H, atoms::C<ops::P>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 0);

    using base_t::base_t;

    static constexpr auto name() noexcept {
        return "QFT2CrAndHadamard"sv;
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        auto last = std::size(qubits) - 1;

        for (std::size_t i(0); i < std::size(qubits); i++) {
            atom<ops::H>()->apply(circuit, ops::H{}, {qubits[last - i]});

            for (std::size_t j = 0; j < std::size(qubits) - 1 - i; ++j) {
                atom<atoms::C<ops::P>>()->apply(circuit, ops::P{1. / static_cast<double>((1UL << (1 + j)))},
                                                {qubits[last - (j + i + 1)], qubits[last - i]});
            }
        }
    }
};
}  // namespace mindquantum::decompositions::rules

#endif /* DECOMPOSITION_RULE_QFT2CRANDHADAMARD_HPP */
