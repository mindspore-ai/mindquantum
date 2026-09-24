/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_PROGRAM_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_PROGRAM_H_

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "simulator/vector/ascend/device_program.h"

namespace mindquantum::sim::ascend {
struct TileProgram {
    uint64_t external_mask;
    uint64_t tile_count;
    uint32_t size;
    uint32_t run;
    std::vector<uint64_t> offsets;
    std::vector<Instruction> instructions;
};

struct DenseProgram {
    uint32_t dimension;
    // Row-major realification [[Re(U), -Im(U)], [Im(U), Re(U)]].
    std::vector<float> coefficients;
    DensePacket packet;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> indices;
};

struct Program {
    qbit_t n_qubits;
    std::vector<std::variant<TileProgram, DenseProgram>> tiles;
    // All Vector tiles address one immutable, program-owned constant pool.
    std::vector<uint32_t> data;
};

struct AdjointTileProgram : TileProgram {
    std::vector<AdjointStep> steps;
    std::vector<AdjointDerivative> derivatives;
    std::vector<GradientTarget> targets;
};

struct DenseAdjointProgram {
    DenseProgram evolution;
    // One realified dB^dagger per active parameter, already Jacobian-weighted.
    std::vector<float> derivatives;
    std::vector<uint32_t> outputs;
    // Restore canonical layout before crossing an adjoint block boundary.
    std::vector<uint32_t> inverse_indices;
};

struct AdjointProgram {
    qbit_t n_qubits;
    uint32_t output_size;
    // Name -> zero-based parameter column. Device output 0 is the overlap;
    // a parameter's device output is 1 + parameter_map.at(name).
    MST<size_t> parameter_map;
    std::vector<std::variant<AdjointTileProgram, DenseAdjointProgram>> tiles;
    std::vector<uint32_t> data;
};

// Shared validation for gate expressions, binding inputs and complete gradient
// batches. Never let a native entry point silently discard imaginary values.
void ValidateResolver(const parameter::ParameterResolver& resolver);

// Own deep snapshots of the gates/PRs/matrices, and plan topology only once.
// Custom callback code must outlive the plan (as for the source CustomGate).
// Bind re-synthesizes coefficients in double, but never calls Planner.
class ForwardPlan {
 public:
    ForwardPlan(const std::vector<std::shared_ptr<BasicGate>>& circuit, qbit_t n_qubits, unsigned tile_qubits,
                unsigned fusion_qubits, unsigned cube_min_gates);
    Program Bind(const parameter::ParameterResolver& parameters) const;

 private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

class AdjointPlan {
 public:
    // Pass the ORIGINAL FORWARD circuit, not its dagger/reversal. Internally
    // it is reversed and lowered using Builder::AdjointGates::Reversed.
    // Requests up to 12 tile bits are accepted; adjoint UB width is capped at
    // 11 bits and fusion capacity at width-3 for aligned low-bit DMA.
    AdjointPlan(const std::vector<std::shared_ptr<BasicGate>>& circuit, const MST<size_t>& parameter_map,
                qbit_t n_qubits, unsigned tile_qubits = 11, unsigned fusion_qubits = 8, unsigned cube_min_gates = 4);
    AdjointProgram Bind(const parameter::ParameterResolver& parameters) const;

 private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

// Pure compilation: validate the entire circuit and snapshot all parameters
// before any device allocation or state mutation. No backend-specific gate math.
Program Compile(const std::vector<std::shared_ptr<BasicGate>>& circuit, const parameter::ParameterResolver& parameters,
                qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits, unsigned cube_min_gates);
AdjointProgram CompileAdjoint(const std::vector<std::shared_ptr<BasicGate>>& circuit,
                              const parameter::ParameterResolver& parameters, const MST<size_t>& parameter_map,
                              qbit_t n_qubits, unsigned tile_qubits = 11, unsigned fusion_qubits = 8);

// Direct apply_gate(diff=True): dU/d(intrinsic angle), NOT a Jacobian-weighted
// parameter gradient. Builder validates exactly one intrinsic parameter, even
// for constant/no_grad gates. Mixing/support-projection operands must fit the
// aligned UB tile. External parity is retained, and inactive external-control
// sectors are explicitly zeroed. CustomGate requires a symbolic parameter
// (which may be no_grad): fixed custom matrices expose no callback-presence API.
Program CompileDerivative(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& parameters,
                          qbit_t n_qubits, unsigned tile_qubits = 12);
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_PROGRAM_H_
