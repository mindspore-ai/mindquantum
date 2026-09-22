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
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_DEVICE_PROGRAM_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_DEVICE_PROGRAM_H_

#include <stdint.h>

namespace mindquantum::sim::ascend {
// Private host/device ABI. Offsets are in 32-bit words, indices in bytes.
// One definition is compiled by both GCC and the Ascend C compiler.
constexpr uint32_t kMaxTileSize = 4096;
constexpr uint32_t kMaxMatrixDimension = 64;
// Each active SIMD row is padded to eight lanes; inactive bases follow it.
constexpr uint32_t kMatrixPadding = 7 * kMaxMatrixDimension;
constexpr uint32_t kStagingSize = kMaxTileSize + kMatrixPadding;
constexpr uint32_t kMaxMatrixBlocks = 4;
constexpr uint32_t kDenseTileSize = 4096;
constexpr uint32_t kDenseBatchPackets = 8;
constexpr uint32_t kDensePipelineSlots = 2;
constexpr uint32_t kMaxDenseQubits = 6;
constexpr uint32_t kMatrix = 0;
constexpr uint32_t kDiagonal = 1;
constexpr uint32_t kStridedMatrix = 2;
constexpr uint32_t kPermutation = 3;
// A derivative instruction is a linear map with ZERO, not identity, on
// unmentioned bases. Its external predicate is still tested by the caller.
constexpr uint32_t kZeroUnmentioned = 0x80000000U;
// Three complex tiles plus executor/reduction scratch fit in the 910B UB.
constexpr uint32_t kMaxAdjointTileSize = 2048;
// Each block owns output_size aligned slots; lanes 0/1 are real/imaginary,
// lanes 2/3 their Kahan errors. Host double-sums (0-2) and (1-3).
constexpr uint32_t kGradientPartialStride = 8;

constexpr uint64_t AdjointPartialBytes(uint32_t blocks, uint32_t output_size) {
    return uint64_t{blocks} * output_size * kGradientPartialStride * sizeof(float);
}

struct AdjointStep {
    uint32_t evolution_begin;
    uint32_t evolution_count;
    uint32_t derivative_begin;
    uint32_t derivative_count;
};
struct AdjointDerivative {
    // Each instruction is an independent, zero-extended piece of dU^dagger.
    // Sum their contractions; never compose these instructions as gates.
    uint32_t instruction_begin;
    uint32_t instruction_count;
    uint32_t target_begin;
    uint32_t target_count;
};
struct GradientTarget {
    uint32_t output_index;
    float real;
    float imag;
    uint32_t reserved;
};
static_assert(sizeof(AdjointStep) == 16 && sizeof(AdjointDerivative) == 16 && sizeof(GradientTarget) == 16,
              "Ascend adjoint metadata ABI mismatch");

// Every control and diagonal operand is already inside the dense operator.
// External bits label independent packets, never an omitted predicate.
struct DensePacket {
    uint64_t plane;
    uint64_t external_mask;
    uint32_t size;
    uint32_t run;
};
static_assert(sizeof(DensePacket) == 24, "Ascend dense-packet ABI mismatch");

// Two disjoint rows of vector repeats. Addresses are float-element offsets;
// strides use the hardware's 32-byte blocks. No amplitude index table is needed.
struct alignas(32) PairSpan {
    uint32_t lhs;
    uint32_t rhs;
    uint32_t mask;
    uint32_t repeats;
    uint32_t stride;
    uint32_t reserved[3];
};
static_assert(sizeof(PairSpan) == 32, "Ascend pair-span ABI mismatch");

struct alignas(32) Instruction {
    uint64_t controls;
    uint64_t parity_mask;
    uint32_t parity;
    uint32_t kind;
    uint32_t dimension;
    uint32_t groups;
    uint32_t forward;
    uint32_t inverse;
    uint32_t coefficients;
    uint32_t staging_size;
    // Interleaved row-major coefficients: one 4x4 or up to four disjoint 2x2
    // blocks. Larger matrices use the constant pool at coefficients instead.
    // Unused block counts are zero; groups is the sum of all counts.
    float matrix[32];
    uint32_t block_groups[kMaxMatrixBlocks];  // SIMD rows, or PairSpan counts
};
static_assert(sizeof(Instruction) == 192, "Ascend instruction ABI mismatch");
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_DEVICE_PROGRAM_H_
