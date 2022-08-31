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

#ifndef CORE_TYPES_HPP
#define CORE_TYPES_HPP

#include <vector>

#include <tweedledum/IR/Cbit.h>
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/IR/Operator.h>
#include <tweedledum/IR/Qubit.h>

namespace mindquantum {
using qubit_id_t = uint32_t;
using qubit_ids_t = std::vector<qubit_id_t>;

using qubit_t = tweedledum::Qubit;
using qubits_t = std::vector<qubit_t>;
using cbit_t = tweedledum::Cbit;
using cbits_t = std::vector<cbit_t>;

using operator_t = tweedledum::Operator;
using instruction_t = tweedledum::Instruction;
using inst_ref_t = tweedledum::InstRef;
using circuit_t = tweedledum::Circuit;
}  // namespace mindquantum

#endif /* CORE_TYPES_HPP */
