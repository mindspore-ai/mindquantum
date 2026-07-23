/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INCLUDE_SIMULATOR_VECTOR_DETAIL_CIRCUIT_TILING_EXECUTOR_H_
#define INCLUDE_SIMULATOR_VECTOR_DETAIL_CIRCUIT_TILING_EXECUTOR_H_

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#include "config/openmp.h"
#include "simulator/vector/circuit_tiling.h"
#include "simulator/vector/detail/circuit_tiling_kernels.h"

namespace mindquantum::sim::vector::detail::circuit_tiling {
template <typename policy_t, typename runtime_t>
void ApplyTile(typename policy_t::qs_data_p_t* state_pointer, const PreparedTile<typename policy_t::py_qs_data_t>& tile,
               index_t dim) {
    using kernels_t = Kernels<typename runtime_t::algebra_t>;
    assert(state_pointer != nullptr);
    assert(dim >= policy_t::DimTh);
    assert(tile.layout.dimension != 0 && tile.layout.state_offsets.size() == tile.layout.dimension);
    assert(tile.layout.tile_count == dim / tile.layout.dimension);
    assert(!tile.operations.empty());

    auto& state = *state_pointer;
    if (state == nullptr) {
        state = policy_t::InitState(dim);
    }
    const auto traversal = runtime_t::MakeTraversal(tile.layout);
    const auto scratch_size = tile.layout.dimension + tile.layout.workspace_size;
    MQ_DO_PRAGMA(omp parallel) {
        auto* scratch = runtime_t::ThreadScratch(scratch_size);
        auto* local_state = scratch;
        auto* workspace = runtime_t::Offset(scratch, tile.layout.dimension);
        runtime_t::ForEachInParallel(
            traversal, tile.layout.tile_count, [&](auto active, auto state_base, const auto& index_context) {
                runtime_t::Gather(state, tile.layout, active, state_base, index_context, local_state);
                for (const auto& operation : tile.operations) {
                    kernels_t::Apply(operation, active, state_base, index_context, local_state, workspace);
                }
                runtime_t::Scatter(state, tile.layout, active, state_base, index_context, local_state);
            });
    }
}

template <typename policy_t, typename runtime_t>
void ApplyAdjointProgram(typename policy_t::qs_data_p_t* shared_state_pointer,
                         const std::vector<typename policy_t::qs_data_p_t*>& batch_state_pointers,
                         const AdjointProgram<typename policy_t::py_qs_data_t>& program, index_t dim,
                         VVT<typename policy_t::py_qs_data_t>* output, bool conjugate_output) {
    using value_t = typename policy_t::py_qs_data_t;
    using kernels_t = Kernels<typename runtime_t::algebra_t>;
    assert(shared_state_pointer != nullptr && output != nullptr);
    assert(dim >= policy_t::DimTh);
    assert(!batch_state_pointers.empty() && !program.tiles.empty() && program.output_size != 0);
    assert(output->size() == batch_state_pointers.size());

    auto& shared_state = *shared_state_pointer;
    if (shared_state == nullptr) {
        shared_state = policy_t::InitState(dim);
    }
    std::vector<typename policy_t::qs_data_p_t> batch_states;
    batch_states.reserve(batch_state_pointers.size());
    for (size_t batch = 0; batch < batch_state_pointers.size(); ++batch) {
        assert(batch_state_pointers[batch] != nullptr && (*output)[batch].size() == program.output_size);
        if (*batch_state_pointers[batch] == nullptr) {
            *batch_state_pointers[batch] = policy_t::InitState(dim);
        }
        batch_states.push_back(*batch_state_pointers[batch]);
    }

    std::vector<typename runtime_t::traversal_t> traversals;
    traversals.reserve(program.tiles.size());
    for (const auto& tile : program.tiles) {
        assert(tile.layout.dimension != 0 && tile.layout.state_offsets.size() == tile.layout.dimension);
        assert(tile.layout.tile_count == dim / tile.layout.dimension && !tile.operations.empty());
        traversals.push_back(runtime_t::MakeTraversal(tile.layout));
    }

    const auto local_output_size = batch_states.size() * program.output_size;
    std::mutex output_mutex;
    // A single OpenMP region spans the whole adjoint program.  Each omp-for's
    // implicit barrier orders consecutive tiles without rebuilding the team.
    MQ_DO_PRAGMA(omp parallel) {
        std::vector<value_t> local_output(local_output_size, 0);
        for (size_t tile_index = 0; tile_index < program.tiles.size(); ++tile_index) {
            const auto& tile = program.tiles[tile_index];
            const auto scratch_size = (1 + batch_states.size()) * tile.layout.dimension + tile.layout.workspace_size;
            auto* scratch = runtime_t::ThreadScratch(scratch_size);
            auto* shared_tile = scratch;
            auto* workspace = runtime_t::Offset(scratch, (1 + batch_states.size()) * tile.layout.dimension);

            runtime_t::ForEachInParallel(
                traversals[tile_index], tile.layout.tile_count,
                [&](auto active, auto state_base, const auto& index_context) {
                    runtime_t::Gather(shared_state, tile.layout, active, state_base, index_context, shared_tile);
                    for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                        runtime_t::Gather(batch_states[batch], tile.layout, active, state_base, index_context,
                                          runtime_t::Offset(scratch, (1 + batch) * tile.layout.dimension));
                    }

                    if (program.compute_overlap && tile_index == 0) {
                        for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                            const auto* batch_tile = runtime_t::Offset(scratch, (1 + batch) * tile.layout.dimension);
                            auto overlap = kernels_t::InnerProduct(active, shared_tile, batch_tile,
                                                                   tile.layout.dimension);
                            local_output[batch * program.output_size] += conjugate_output ? std::conj(overlap)
                                                                                          : overlap;
                        }
                    }

                    for (const auto& operation : tile.operations) {
                        kernels_t::Apply(operation.evolution, active, state_base, index_context, shared_tile,
                                         workspace);
                        for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                            auto* batch_tile = runtime_t::Offset(scratch, (1 + batch) * tile.layout.dimension);
                            for (const auto& derivative : operation.derivatives) {
                                const auto intrinsic = kernels_t::ExpectDerivative(
                                    derivative, active, state_base, index_context, shared_tile, batch_tile);
                                for (const auto& target : derivative.targets) {
                                    auto contribution = intrinsic * target.coefficient;
                                    if (conjugate_output) {
                                        contribution = std::conj(contribution);
                                    }
                                    local_output[batch * program.output_size + target.output_index] += contribution;
                                }
                            }
                            kernels_t::Apply(operation.evolution, active, state_base, index_context, batch_tile,
                                             workspace);
                        }
                    }

                    runtime_t::Scatter(shared_state, tile.layout, active, state_base, index_context, shared_tile);
                    for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                        runtime_t::Scatter(batch_states[batch], tile.layout, active, state_base, index_context,
                                           runtime_t::Offset(scratch, (1 + batch) * tile.layout.dimension));
                    }
                });
        }

        const std::lock_guard<std::mutex> lock(output_mutex);
        for (size_t batch = 0; batch < batch_states.size(); ++batch) {
            for (size_t column = 0; column < program.output_size; ++column) {
                (*output)[batch][column] += local_output[batch * program.output_size + column];
            }
        }
    }
}
}  // namespace mindquantum::sim::vector::detail::circuit_tiling

#endif  // INCLUDE_SIMULATOR_VECTOR_DETAIL_CIRCUIT_TILING_EXECUTOR_H_
