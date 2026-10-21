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
#include <set>
#include <string>
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
    // Instruction offsets address this immutable pool. Static and bound tiles
    // may share different pools without relocating their instructions.
    std::shared_ptr<const std::vector<uint32_t>> data;
};

struct DenseLayout {
    // Read through shared layouts by operator synthesis and Cube placement.
    // Cppcheck 1.90 cannot follow those cross-translation-unit accesses.
    uint32_t dimension;
    DensePacket packet;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> indices;
};

struct DenseProgram {
    std::shared_ptr<const DenseLayout> layout;
    // Row-major realification [[Re(U), -Im(U)], [Im(U), Re(U)]].
    std::vector<float> coefficients;
};

struct Program {
    qbit_t n_qubits;
    std::vector<std::variant<std::shared_ptr<const TileProgram>, std::shared_ptr<const DenseProgram>>> tiles;
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
};

struct AdjointProgram {
    qbit_t n_qubits;
    // Name -> zero-based parameter column. Device output 0 is the overlap;
    // a parameter's device output is 1 + parameter_map.at(name).
    MST<size_t> parameter_map;
    std::vector<std::variant<std::shared_ptr<const AdjointTileProgram>, std::shared_ptr<const DenseAdjointProgram>>>
        tiles;
};

// Shared validation for gate expressions, binding inputs and complete gradient
// batches. Never let a native entry point silently discard imaginary values.
void ValidateResolver(const parameter::ParameterResolver& resolver);
void ValidateConfiguration(qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits);

// One scheduling/binding implementation for both directions. Plans own their
// gate snapshots and fold constant blocks once; Bind returns a complete program.
// An adjoint can share a forward plan's validated input without copying gates.
// Custom callback code must outlive the plan (as for the source CustomGate).
template <typename program_t>
class CircuitPlan {
 public:
    // Always pass the original forward circuit. Adjoint scheduling reverses
    // it internally, caps UB width at 11, and preserves layout across blocks.
    CircuitPlan(const std::vector<std::shared_ptr<BasicGate>>& circuit, qbit_t n_qubits, unsigned tile_qubits,
                unsigned fusion_qubits, unsigned cube_min_gates, const MST<size_t>& parameter_map = {});
    CircuitPlan(const CircuitPlan<Program>& forward, const MST<size_t>& parameter_map, unsigned tile_qubits,
                unsigned fusion_qubits, unsigned cube_min_gates);
    program_t Bind(const parameter::ParameterResolver& parameters) const;
    const std::set<std::string>& Parameters() const;
    const std::set<std::string>& GradientParameters() const;

 private:
    template <typename>
    friend class CircuitPlan;
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

using ForwardPlan = CircuitPlan<Program>;
using AdjointPlan = CircuitPlan<AdjointProgram>;

// Pure compilation: validate the entire circuit and snapshot all parameters
// before any device allocation or state mutation. No backend-specific gate math.
Program Compile(const std::vector<std::shared_ptr<BasicGate>>& circuit, const parameter::ParameterResolver& parameters,
                qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits, unsigned cube_min_gates);

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
