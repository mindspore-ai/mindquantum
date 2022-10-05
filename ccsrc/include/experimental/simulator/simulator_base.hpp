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

#ifndef SIMULATOR_BASE_HPP
#define SIMULATOR_BASE_HPP

#include <cstdint>

#include "experimental/core/circuit_block.hpp"
#include "experimental/core/types.hpp"
#include "experimental/simulator/config.hpp"
#if MQ_HAS_CONCEPTS
#    include "experimental/simulator/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::simulation {
//! A base class for all simulators
template <typename derived_t>
class BaseSimulator {
 public:
    using base_t = BaseSimulator;
    using parent_t = BaseSimulator<derived_t>;

    //! Simple constructor
    /*!
     * \param seed Random seed to initialiue the random number generator.
     */
    explicit BaseSimulator(uint32_t seed = 0);

    // -------------------------------------------------------------------------
    // Methods that need to be implemented in the child class
    //
    // bool has_qubit(const qubit_t& qubit);
    // bool allocate_qubits(const qubits_t& qubits);
    // bool run_instruction(const instruction_t& inst);

    // =============================================================================

    //! Run a quantum circuit using a simulator
    /*!
     * \param circuit A quantum circuit
     *
     * \note This function will call
     */
#if MQ_HAS_CONCEPTS
    template <concepts::CircuitLike circuit_like_t>
#else
    template <typename circuit_like_t>
#endif  // MQ_HAS_CONCEPTS
    MQ_NODISCARD bool run_circuit(const circuit_like_t& circuit);

 private:
    uint32_t seed_;
};
}  // namespace mindquantum::simulation

#include "simulator_base.tpp"

#endif /* SIMULATOR_BASE_HPP */
