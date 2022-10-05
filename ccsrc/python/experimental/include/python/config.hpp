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

#ifndef PYTHON_CONFIG_HPP
#define PYTHON_CONFIG_HPP

#include <vector>

#include <tweedledum/IR/Cbit.h>
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/IR/Qubit.h>

#include "experimental/core/config.hpp"

namespace mindquatum::python {
using circuit_t = tweedledum::Circuit;
using instruction_t = tweedledum::Instruction;
using operator_t = tweedledum::Operator;
using qubit_t = tweedledum::Qubit;
using qubits_t = std::vector<qubit_t>;
using cbit_t = tweedledum::Cbit;
using cbits_t = std::vector<cbit_t>;
}  // namespace mindquatum::python

#endif /* PYTHON_CONFIG_HPP */
