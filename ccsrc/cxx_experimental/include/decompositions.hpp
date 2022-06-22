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

#ifndef DECOMPOSITIONS_HPP
#define DECOMPOSITIONS_HPP
#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Operators/Standard/P.h>

namespace mindquantum::decompositions {
using circuit_t = tweedledum::Circuit;
using instruction_t = tweedledum::Instruction;

void decompose_cnot2cz(circuit_t& result, const instruction_t& inst);
//! Decompose CNOT gate, M for 'Minus' because ends with Ry(-pi/2)
void decompose_cnot2rxx_M(circuit_t& result, const instruction_t& inst);
//! Decompose CNOT gate, M for 'Minus' because ends with Ry(+pi/2)
void decompose_cnot2rxx_P(circuit_t& result, const instruction_t& inst);

//! Decompose multi-controlled gates
void decompose_cnu2toffoliandcu(circuit_t& result, const instruction_t& inst);

//! Decompose entangle into h and cnot/cx gates
void decompose_entangle(circuit_t& result, const instruction_t& inst);

//! Decompose controlled global phase gates into (controlled) R/r1
/*!
 * Shaves off one control qubit when applied
 */
void decompose_ph2r(circuit_t& result, const instruction_t& inst);

//! Delete all phase/ph gates without controls
void decompose_PhNoCtrl(circuit_t& result, const instruction_t& inst);

//! Decompose H gate, M for 'Minus' because ends with Ry(-pi/2)
void decompose_h2rx_M(circuit_t& result, const instruction_t& inst);
//! Decompose H gate, N for 'Neutral'
void decompose_h2rx_N(circuit_t& result, const instruction_t& inst);

//! Decompose QFT (Quantum Fourier Transform) into h and controlled R (cr1)
void decompose_qft2crandhadamard(circuit_t& result, const instruction_t& inst);

//! Decompose (controlled) Qubit Operator into (controlled) Paulis
void decompose_qubitop2onequbit(circuit_t& result, const instruction_t& inst);

void decompose_rx2rz(circuit_t& result, const instruction_t& inst);
void decompose_ry2rz(circuit_t& result, const instruction_t& inst);

void decompose_rz2rx_P(circuit_t& result, const instruction_t& inst);
void decompose_rz2rx_M(circuit_t& result, const instruction_t& inst);

//! Decompose (controlled) phase-shift gate using z-rotation and global phase
void decompose_r2rzandph(circuit_t& result, const instruction_t& inst);

//! Decompose sqrtswap into controlled x and sqrtx
void decompose_sqrtswap2cnot(circuit_t& result, const instruction_t& inst);

//! Decompose swap into controlled x gates
void decompose_swap2cnot(circuit_t& result, const instruction_t& inst);

bool recognize_time_evolution_commuting(const instruction_t& inst);
void decompose_time_evolution_commuting(circuit_t& result, const instruction_t& inst);
bool recognize_time_evolution_individual_terms(const instruction_t& inst);
void decompose_time_evolution_individual_terms(circuit_t& result, const instruction_t& inst);

//! Decompose toffoli gate (ccx) into cx, t, tdg and h gates
void decompose_toffoli2cnotandtgate(circuit_t& result, const instruction_t& inst);
}  // namespace mindquantum::decompositions
#endif /* DECOMPOSITIONS_HPP */
