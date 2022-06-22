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
//   See the License for te specific language governing permissions and
//   limitations under the License.

#ifndef PYTHON_OPTIMIZTE_HPP
#define PYTHON_OPTIMIZTE_HPP

#include <tweedledum/Passes/Optimization/gate_cancellation.h>
#include <tweedledum/Passes/Optimization/phase_folding.h>

#include "core/config.hpp"

#if MQ_HAS_CONCEPTS
#    include "core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

namespace mindquantum::python {
class GateCancellation {
 public:
#if MQ_HAS_CONCEPTS
    template <concepts::CircuitLike circuit_like_t>
#else
    template <typename circuit_like_t>
#endif  // MQ_HAS_CONCEPTS
    MQ_NODISCARD bool run_circuit(circuit_like_t& circuit) const noexcept;
};
}  // namespace mindquantum::python

#include "python/optimizer/optimizer.tpp"

#endif /* PYTHON_OPTIMIZTE_HPP */
