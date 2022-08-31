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

#ifndef SIMULATOR_CONCEPTS_HPP
#define SIMULATOR_CONCEPTS_HPP

#include "experimental/core/types.hpp"
#include "experimental/simulator/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::concepts {
#if MQ_HAS_CONCEPTS
//! C++20 concept representing a basic simulator API.
template <typename simulator_t>
concept CircuitSimulator = requires(simulator_t simulator, qubit_t qubit, qubits_t qubits, instruction_t inst) {
    { simulator.run_instruction(inst) } -> same_decay_as<bool>;
    { simulator.allocate_qubits(qubits) } -> same_decay_as<bool>;
    { simulator.has_qubit(qubit) } -> same_decay_as<bool>;
};
#endif  // MQ_HAS_CONCEPTS
}  // namespace mindquantum::concepts

#endif /* SIMULATOR_CONCEPTS_HPP */
