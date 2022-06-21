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

#ifndef PYTHON_OPTIMIZER_HPP
#define PYTHON_OPTIMIZER_HPP

#ifndef PYTHON_OPTIMIZER_HPP
#    error This file must only be included by python/!
#endif  // PYTHON_OPTIMIZER_HPP

// NB: This is mainly for syntax checkers and completion helpers as this file
//     is only intended to be included directly by python/
#include "python/optimizer/optimizer.hpp"

namespace mindquantum::python {
#if MQ_HAS_CONCEPTS
template <concepts::CircuitLike circuit_like_t>
#else
template <typename circuit_like_t>
#endif  // MQ_HAS_CONCEPTS
bool GateCancellation::run_circuit(circuit_like_t& circuit) const noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<circuit_like_t>, std::remove_cvref_t<circuit_t>>) {
        circuit = tweedledum::gate_cancellation(circuit);
        circuit = tweedledum::phase_folding(circuit);
    } else {
        circuit.transform(tweedledum::gate_cancellation);
        circuit.transform(tweedledum::phase_folding);
    }
    return true;
}
}  // namespace mindquantum::python

#endif /* PYTHON_OPTIMIZER_HPP */
