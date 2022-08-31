//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef DECOMPOSITIONS_CONFIG_HPP
#define DECOMPOSITIONS_CONFIG_HPP

#include <cstdint>
#include <vector>

#include <tweedledum/IR/Cbit.h>
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/IR/Qubit.h>

#include "experimental/core/config.hpp"

namespace mindquantum::decompositions {
using num_target_t = uint32_t;
using num_control_t = int32_t;
using num_param_t = uint32_t;

//! Constant representing no constraints on the number of target qubits
static constexpr auto any_target = num_target_t(0);

//! Constant representing no constraints on the number of control qubits
static constexpr auto any_control = num_control_t(-1);
}  // namespace mindquantum::decompositions

namespace mindquantum::traits {
//! Type traits class to calculate control qubit related constants
template <decompositions::num_control_t num_controls_>
struct controls {
    /*!
     * Number of controls qubits to be used when decomposing a gate. This constant is required to calculate which
     * control qubits are "free" control qubits and which ones are actually required as part of the decomposition
     * rule of a gate.
     *
     * e.g. (a) the X -> H Z H decomposition has no constraint -> 0
     *      (b) the CX -> H CZ H decomposition is constrained to 1 control qubit -> 1
     *
     *      -> applying (a) to CCX would therefore have 2 "free" control qubits
     *      -> applying (b) to CCX would therefore have 1 "free" control qubit and 1 control qubit used in the
     *         decomposition rule
     */
    static constexpr auto num_controls_for_decomp = (num_controls_ == decompositions::any_control)
                                                        ? decompositions::num_control_t(0)
                                                        : num_controls_;
};
}  // namespace mindquantum::traits

#endif /* DECOMPOSITIONS_CONFIG_HPP */
