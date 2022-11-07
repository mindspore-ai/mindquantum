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

#include "experimental/ops/gates/time_evolution.hpp"

#include <tweedledum/Operators/Standard/H.h>
#include <tweedledum/Operators/Standard/Rx.h>
#include <tweedledum/Operators/Standard/Ry.h>
#include <tweedledum/Operators/Standard/Rz.h>
#include <tweedledum/Operators/Standard/X.h>

#include "ops/gates/qubit_operator.hpp"

#include "experimental/core/compute.hpp"
#include "experimental/core/dagger.hpp"
#include "experimental/decompositions.hpp"

#ifndef M_PI
#    define M_PI 3.14159265358979323846 /* pi */
#endif

namespace mindquantum::decompositions {
namespace td = tweedledum;

bool recognize_time_evolution_commuting(const instruction_t& inst) {
    assert(inst.kind() == "projectq.timeevolution");

    const auto& op = inst.cast<ops::TimeEvolution>();

    auto terms = op.get_hamiltonian().get_terms();
    if (std::size(terms) == 1) {
        return false;
    } else {
        const auto num_targets = op.num_targets();
        for (const auto& term : terms) {
            const ops::QubitOperator<std::complex<double>> test_op(term.first, term.second);
            for (const auto& other : terms) {
                const ops::QubitOperator<std::complex<double>> other_op(other.first, other.second);
                const auto& commutator = test_op * other_op - other_op * test_op;
                if (!commutator.is_identity(1.e-9)) {
                    return false;
                }
            }
        }
        return true;
    }
}

void decompose_time_evolution_commuting(circuit_t& result, const instruction_t& inst) {  // NOLINT
    assert(inst.kind() == "projectq.timeevolution");
    assert(recognize_time_evolution_commuting(inst));

    const auto& time_evolution(inst.cast<ops::TimeEvolution>());

    auto time = time_evolution.get_time();
    const auto& hamiltonian = time_evolution.get_hamiltonian();
    // auto num_targets = time_evolution.num_targets();
    const auto& qubits = inst.qubits();

    for (const auto& term : hamiltonian.get_terms()) {
        ops::QubitOperator<std::complex<double>> ind_operator(term.first, term.second);
        decompose_time_evolution_individual_terms(result,
                                                  td::Instruction(ops::TimeEvolution(ind_operator, time), qubits, {}));

        // result.apply_operator(ops::TimeEvolution(ind_operator, time), qubits);
    }
}

bool recognize_time_evolution_individual_terms(const instruction_t& inst) {
    assert(inst.kind() == "projectq.timeevolution");

    const auto& op = inst.cast<ops::TimeEvolution>();
    auto terms = op.get_hamiltonian().get_terms();
    return std::size(terms) == 1;
}

namespace impl {
template <typename CircuitType>
void decompose_time_evolution_individual_terms(CircuitType& result, const instruction_t& inst) {  // NOLINT
    assert(inst.kind() == "projectq.timeevolution");
    assert(recognize_time_evolution_individual_terms(inst));

    const auto& time_evolution(inst.cast<ops::TimeEvolution>());
    auto time = time_evolution.get_time();
    const auto& hamiltonian = time_evolution.get_hamiltonian();

    const auto& term = std::begin(hamiltonian.get_terms())->first;
    const auto& coefficient = std::abs(std::begin(hamiltonian.get_terms())->second);
    // TODO(dnguyen): Make imaginary time possible!
    auto qubits = inst.qubits();
    decltype(qubits) targets(std::end(qubits) - inst.num_targets(), std::end(qubits));
    assert(std::size(targets) > 0 && "TimeEvolution must have targets");

    // Restrict qubits to control qubits only
    qubits.resize(std::size(qubits) - inst.num_targets(), td::Qubit::invalid());

    // Sanity check
    assert(std::size(qubits) + std::size(targets) == inst.num_wires());

    if (std::size(term) == 1) {
        // hamiltonian has only a single local operator

        qubits.push_back(targets[term[0].first]);

        if (term[0].second == ops::TermValue::X) {
            result.apply_operator(td::Op::Rx(time * coefficient * 2.), qubits, {});
        } else if (term[0].second == ops::TermValue::Y) {
            result.apply_operator(td::Op::Ry(time * coefficient * 2.), qubits, {});
        } else {
            // NB: missing * 2 factor due to Tweedledum Rz definition
            result.apply_operator(td::Op::Rz(time * coefficient), qubits, {});
        }
    } else {
        // hamiltonian has more than one local operator
        MQ_WITH_COMPUTE(result, circuit) {
            for (const auto& [index, action] : term) {
                qubits.push_back(targets[index]);
                if (action == ops::TermValue::X) {
                    circuit.apply_operator(td::Op::H(), qubits);
                } else if (action == ops::TermValue::Y) {
                    circuit.apply_operator(td::Op::Rx(0.5), qubits);
                }
                qubits.pop_back();
            }

            // Compute parity
            for (std::size_t i = 0; i < std::size(targets) - 1; ++i) {
                qubits.push_back(targets[i]);
                qubits.push_back(targets[i + 1]);
                circuit.apply_operator(td::Op::X(), qubits);
                qubits.pop_back();
                qubits.pop_back();
            }
        }
        MQ_WITH_COMPUTE_END

        qubits.push_back(targets.back());
        // NB: missing * 2 factor due to Tweedledum Rz definition
        circuit.apply_operator(td::Op::Rz(time * coefficient), qubits);
        qubits.pop_back();

        // Automatic uncompute
    }
}
}  // namespace impl

void decompose_time_evolution_individual_terms(circuit_t& result, const instruction_t& inst) {  // NOLINT
    assert(inst.kind() == "projectq.timeevolution");

    impl::decompose_time_evolution_individual_terms(result, inst);
}
}  // namespace mindquantum::decompositions
