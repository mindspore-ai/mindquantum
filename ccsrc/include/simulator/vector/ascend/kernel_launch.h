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
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_LAUNCH_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_LAUNCH_H_

#include <stdint.h>

#include <acl/acl.h>

#include "simulator/vector/ascend/device_program.h"

namespace mindquantum::sim::ascend {
aclError LaunchTile(uint32_t blocks, aclrtStream stream, void* state, void* instructions, void* data, void* offsets,
                    uint64_t plane, uint64_t external_mask, uint64_t tiles, uint32_t size, uint32_t run,
                    uint32_t count);
aclError LaunchReset(uint32_t blocks, aclrtStream stream, void* state, uint64_t plane);
aclError LaunchDense(uint32_t blocks, aclrtStream stream, void* state, void* matrix, void* offsets, void* indices,
                     void* tiling, void* workspace, DensePacket packet);

// Launch every AdjointProgram tile in order on the SAME stream and with the
// SAME positive block count (idle blocks are allowed). Both states are distinct
// canonical SoA FP32 buffers of 2*plane floats, plane=max(8,2^n_qubits).
// Upload tile.instructions/steps/derivatives/targets/offsets and program.data
// unchanged. Empty metadata arrays may be null. No extra device workspace.
// size<=kMaxAdjointTileSize, size/run powers of two, 8<=run<=size<=plane.
// Zero AdjointPartialBytes(blocks, output_size) bytes ONCE before the sweep;
// partials are read/added/written, never atomically shared between blocks.
// compute_overlap=1 only for the first tile, including an empty circuit's tile.
// After stream completion, read only partials. For each output j, its slot is
// (block*output_size+j)*kGradientPartialStride. Sum lanes (0-2) and (1-3),
// converting each operand to DOUBLE before subtracting on the host.
//
// Let shared=|psi_after>, costate=|lambda_after>. Per gate we compute
// psi_before=U^dagger psi_after, then <psi_before|dU^dagger|lambda_after>,
// weighted by Builder's (unconjugated) Jacobian targets, then evolve lambda.
// NO output conjugation is performed. For real parameter maps and costate
// H|psi>, this returns g_H=<dpsi|H|psi>, and output[0]=<psi|H|psi>.
// Hermitian derivative: 2*Re(g_H). Non-Hermitian derivative from two sweeps:
// g_H + conj(g_Hdagger). For different left/right circuits use the appropriate
// left- and right-state sweeps, conjugating the right sweep's result.
aclError LaunchAdjointTile(uint32_t blocks, aclrtStream stream, void* shared_state, void* costate, void* instructions,
                           void* data, void* offsets, void* steps, void* derivatives, void* targets, void* partials,
                           uint64_t plane, uint64_t external_mask, uint64_t tiles, uint32_t size, uint32_t run,
                           uint32_t step_count, uint32_t output_size, uint32_t compute_overlap);

// The same contraction for a fused block B. Cube evolves BOTH states by B^dagger,
// then contracts <psi_before|dB^dagger|lambda_after> for each active parameter.
// Derivatives are already Jacobian-weighted. Routing is restored before return.
// blocks is the Cube count; its 2*blocks AIVs must match the Vector sweep's slot
// ownership. Workspace is the library slice plus 10*packet.size*batch_packets
// floats per AIV, independent of state size and the number of derivatives.
aclError LaunchDenseAdjoint(uint32_t blocks, aclrtStream stream, void* shared_state, void* costate, void* matrix,
                            void* derivatives, void* outputs, void* offsets, void* indices, void* inverse_indices,
                            void* tiling, void* workspace, void* partials, DensePacket packet,
                            uint32_t derivative_count, uint32_t output_size, uint32_t compute_overlap);
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_KERNEL_LAUNCH_H_
