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

#ifndef DECOMPOSITION_RULE_CNOT2RXX_HPP
#define DECOMPOSITION_RULE_CNOT2RXX_HPP

#include <tuple>

#include "experimental/decompositions/gate_decomposition_rule.hpp"
#include "experimental/decompositions/rules/config.hpp"
#include "experimental/ops/gates.hpp"
#include "experimental/ops/parametric/angle_gates.hpp"

namespace mindquantum::decompositions::rules {
class CNOT2Rxx
    : public GateDecompositionRule<CNOT2Rxx, std::tuple<ops::X>, SINGLE_TGT_SINGLE_CTRL, ops::parametric::Rx,
                                   ops::parametric::Ry, ops::parametric::Ph, atoms::C<ops::parametric::Rxx>> {
 public:
    static_assert(self_t::num_controls_for_decomp == 1);

    using base_t::base_t;

    explicit CNOT2Rxx(AtomStorage& storage) : base_t{storage}, use_positive_decomp_{false} {
    }

    static constexpr auto name() noexcept {
        return "CNOT2Rxx"sv;
    }

    void apply_positive_decomp(circuit_t& circuit, const qubits_t& qubits) {
        using ops::parametric::Ph;
        using ops::parametric::Rx;
        using ops::parametric::Rxx;
        using ops::parametric::Ry;

        atom<Ry>()->apply(circuit, ops::Ry{-PI_VAL_2}, {qubits[0]});
        atom<Ph>()->apply(circuit, ops::Ph{PI_VAL_4}, {qubits[0]});
        atom<Rx>()->apply(circuit, ops::Rx{-PI_VAL_2}, {qubits[0]});
        atom<Rx>()->apply(circuit, ops::Rx{PI_VAL_2}, {qubits[1]});
        // NB: pi_half -> pi_quarter compared to ProjectQ because of Tweedledum gate definition
        atom<atoms::C<Rxx>>()->apply(circuit, ops::Rxx{PI_VAL_4}, qubits);
        atom<Ry>()->apply(circuit, ops::Ry{PI_VAL_2}, {qubits[0]});
    }

    void apply_negative_decomp(circuit_t& circuit, const qubits_t& qubits) {
        using ops::parametric::Ph;
        using ops::parametric::Rx;
        using ops::parametric::Rxx;
        using ops::parametric::Ry;

        atom<Ry>()->apply(circuit, ops::Ry{PI_VAL_2}, {qubits[0]});
        atom<Ph>()->apply(circuit, ops::Ph{PI_VAL_4 * 7.}, {qubits[0]});
        atom<Rx>()->apply(circuit, ops::Rx{-PI_VAL_2}, {qubits[0]});
        atom<Rx>()->apply(circuit, ops::Rx{-PI_VAL_2}, {qubits[1]});
        // NB: pi_half -> pi_quarter compared to ProjectQ because of Tweedledum gate definition
        atom<atoms::C<Rxx>>()->apply(circuit, ops::Rxx{PI_VAL_4}, qubits);
        atom<Ry>()->apply(circuit, ops::Ry{-PI_VAL_2}, {qubits[0]});
    }

    void apply_impl(circuit_t& circuit, const operator_t& /* op */, const qubits_t& qubits,
                    const cbits_t& /* unused */) {
        if (use_positive_decomp_) {
            apply_positive_decomp(circuit, qubits);
        } else {
            apply_negative_decomp(circuit, qubits);
        }
        use_positive_decomp_ ^= true;
    }

 private:
    bool use_positive_decomp_;
};
}  // namespace mindquantum::decompositions::rules
#endif /* DECOMPOSITION_RULE_CNOT2RXX_HPP */
