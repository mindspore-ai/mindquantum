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

#include <tweedledum/IR/Qubit.h>
#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>

#include "ops/gates/details/std_complex_coeff_policy.hpp"
#include "ops/gates/qubit_operator.hpp"

#include "experimental/decompositions.hpp"
#include "experimental/ops/gates/ph.hpp"

namespace mindquantum::decompositions {
namespace td = tweedledum;

void decompose_qubitop2onequbit(circuit_t& result, const instruction_t& inst) {  // NOLINT
    auto qubits = inst.qubits();
    const auto& terms = inst.cast<ops::QubitOperator<std::complex<double>>>().get_terms();
    decltype(qubits) targets(std::end(qubits) - inst.num_targets(), std::end(qubits));

    // Only keep control qubits in qubits
    qubits.resize(std::size(qubits) - inst.num_targets(), td::Qubit::invalid());

    qubits.push_back(targets[targets[0]]);
    result.apply_operator(ops::Ph(std::arg(std::begin(terms)->second)), qubits);
    qubits.pop_back();

    for (const auto& pauli : std::begin(terms)->first) {
        // qubits.push_back(targets[pauli.first]);
        // if (pauli.second == 'X') {
        //     result.apply_operator(td::Op::X(), qubits);
        // } else if (pauli.second == 'Y') {
        //     result.apply_operator(td::Op::Y(), qubits);
        // } else if (pauli.second == 'Z') {
        //     result.apply_operator(td::Op::Z(), qubits);
        // } else {
        //     assert(0 && "QubitOperator Pauli must be 'X', 'Y' or 'Z'");
        // }
        // qubits.pop_back();
    }
}
}  // namespace mindquantum::decompositions
