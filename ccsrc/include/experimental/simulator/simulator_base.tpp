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

#ifndef SIMULATOR_BASE_TPP
#define SIMULATOR_BASE_TPP

#ifndef SIMULATOR_BASE_HPP
#    error This file must only be included by simulator/simulator_base.hpp!
#endif  // SIMULATOR_BASE_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by simulator_base.hpp
#include "experimental/simulator/simulator_base.hpp"

namespace mindquantum::simulation {
template <typename derived_t>
BaseSimulator<derived_t>::BaseSimulator(uint32_t seed) : seed_(seed) {
#if MQ_HAS_CONCEPTS
    static_assert(concepts::CircuitSimulator<derived_t>);
#endif  // MQ_HAS_CONCEPTS
}

// =============================================================================

template <typename derived_t>
#if MQ_HAS_CONCEPTS
template <concepts::CircuitLike circuit_like_t>
#else
template <typename circuit_like_t>
#endif  // MQ_HAS_CONCEPTS
bool BaseSimulator<derived_t>::run_circuit(const circuit_like_t& circuit) {
    auto* simulator{static_cast<derived_t*>(this)};

    qubits_t qubits_to_allocate;
    for (const auto& qubit : circuit.qubits()) {
        if (!simulator->has_qubit(qubit)) {
            qubits_to_allocate.push_back(qubit);
        }
    }

    if (!simulator->allocate_qubits(qubits_to_allocate)) {
        return false;
    }

    auto run_ok = true;
    circuit.foreach_instruction([&simulator, &run_ok](const instruction_t& inst) {
        if (run_ok) {
            run_ok &= simulator->run_instruction(inst);
        }
    });

    return run_ok;
}
}  // namespace mindquantum::simulation
#endif /* SIMULATOR_BASE_TPP */
