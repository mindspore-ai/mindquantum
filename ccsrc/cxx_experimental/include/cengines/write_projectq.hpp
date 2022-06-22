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
#ifndef WRITE_PROJECTQ_HPP
#define WRITE_PROJECTQ_HPP

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <kitty/detail/mscfix.hpp>
#include <tweedledum/IR/Circuit.h>

#include "ops/gates.hpp"

namespace td = tweedledum;

namespace mindquantum {
class QubitOperator;
std::string to_string(std::size_t qubit_id);
std::string to_string(const std::vector<std::size_t>& qubit_ids);
std::string to_string(const ops::QubitOperator& qb_op, std::vector<std::size_t> const& targets);
std::string to_string(std::string_view kind);
std::string to_string(const td::Instruction& inst);
}  // namespace mindquantum

namespace mindquantum {
/*! \brief Writes instruction in ProjecQ format into output stream
 *
 * \param inst Instruction to print out
 * \param os Output stream
 */
void write_projectq(const td::Instruction& inst, std::ostream& os = std::cout);

/*! \brief Writes circuit in ProjecQ format into an output stream
 *
 * \param circuit Quantum circuit
 * \param os Output stream
 */
template <typename circuit_t>
void write_projectq(const circuit_t& circuit, std::ostream& os = std::cout) {
    circuit.foreach_instruction([&os](const td::Instruction& inst) { write_projectq(inst, os); });
}

/*! \brief Writes circuit in ProjecQ format into a file
 *
 * \param circuit Circuit
 * \param filename Filename
 */
void write_projectq(const td::Circuit& circuit, std::string_view filename);
}  // namespace mindquantum

#endif /* WRITE_PROJECTQ_HPP */
