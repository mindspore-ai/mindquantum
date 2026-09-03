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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_TILE_ENGINE_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_TILE_ENGINE_H_

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#include "config/openmp.h"
#include "core/utils.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/circuit_tiling/detail/kernels.h"

// The tile engine drives a runtime over a prepared tile program.  A runtime
// supplies the numerical algebra, the thread scratch storage, the traversal
// of tile blocks (serial and OpenMP-parallel), and the gather/scatter between
// the flat state and a cache-resident local block:
//
//   struct Runtime {
//       using algebra_t;    // see detail::Kernels
//       using storage_t;    // element type of local block storage
//       using traversal_t;
//       static constexpr bool kMirrorsHermitianTiles;  // evolves only one block of each mirrored pair
//       static traversal_t MakeTraversal(const TileLayout&);
//       static storage_t* ThreadScratch(size_t logical_size);
//       static storage_t* Offset(storage_t*, size_t logical_offset);
//       static void ForEach(const traversal_t&, const F&);            // serial
//       static void ForEachInParallel(const traversal_t&, const F&);  // inside omp parallel
//       static void Gather(const value_t* state, const TileLayout&, active, state_base, context, storage_t* block);
//       static void Scatter(value_t* state, const TileLayout&, active, state_base, context, const storage_t* block);
//       // Only with kMirrorsHermitianTiles:
//       static bool Mirrored(const context_t&);          // this block's mirror is written on scatter
//       static state_index_t MirrorBase(const context_t&);
//       static void MirrorBlock(const TileLayout&, const storage_t* block, storage_t* mirror);
//   };
namespace mindquantum::sim::circuit_tiling::detail {
template <typename runtime_t, typename value_t>
void ApplyTile(value_t* state, const PreparedTile<value_t>& tile, bool parallel) {
    using kernels_t = Kernels<typename runtime_t::algebra_t>;
    assert(state != nullptr);
    assert(tile.layout.dimension != 0 && tile.layout.state_offsets.size() == tile.layout.dimension);
    assert(!tile.operations.empty());

    const auto traversal = runtime_t::MakeTraversal(tile.layout);
    const auto scratch_size = tile.layout.dimension + tile.layout.workspace_size;
    const auto run = [&](auto* scratch, const auto& for_each) {
        auto* local_state = scratch;
        auto* workspace = runtime_t::Offset(scratch, tile.layout.dimension);
        for_each(traversal, [&](auto active, auto state_base, const auto& index_context) {
            runtime_t::Gather(state, tile.layout, active, state_base, index_context, local_state);
            for (const auto& operation : tile.operations) {
                kernels_t::Apply(operation, active, state_base, index_context, local_state, workspace);
            }
            runtime_t::Scatter(state, tile.layout, active, state_base, index_context, local_state);
        });
    };
    if (!parallel) {
        run(runtime_t::ThreadScratch(scratch_size),
            [](const auto& traversal, const auto& body) { runtime_t::ForEach(traversal, body); });
        return;
    }
    MQ_DO_PRAGMA(omp parallel) {
        run(runtime_t::ThreadScratch(scratch_size),
            [](const auto& traversal, const auto& body) { runtime_t::ForEachInParallel(traversal, body); });
    }
}

template <typename runtime_t, typename value_t>
void ApplyAdjointProgram(value_t* shared_state, const std::vector<value_t*>& batch_states,
                         const AdjointProgram<value_t>& program, VVT<value_t>* output, bool conjugate_output,
                         bool parallel) {
    using kernels_t = Kernels<typename runtime_t::algebra_t>;
    assert(shared_state != nullptr && output != nullptr);
    assert(!batch_states.empty() && !program.tiles.empty() && program.output_size != 0);
    assert(output->size() == batch_states.size());
    assert(std::none_of(batch_states.begin(), batch_states.end(), [](const auto* state) { return state == nullptr; }));

    std::vector<typename runtime_t::traversal_t> traversals;
    traversals.reserve(program.tiles.size());
    for (const auto& tile : program.tiles) {
        assert(tile.layout.dimension != 0 && tile.layout.state_offsets.size() == tile.layout.dimension);
        assert(!tile.operations.empty());
        traversals.push_back(runtime_t::MakeTraversal(tile.layout));
    }

    const auto local_output_size = batch_states.size() * program.output_size;
    std::mutex output_mutex;
    const auto sweep = [&](const auto& for_each) {
        std::vector<value_t> local_output(local_output_size, 0);
        for (size_t tile_index = 0; tile_index < program.tiles.size(); ++tile_index) {
            const auto& tile = program.tiles[tile_index];
            const auto dimension = tile.layout.dimension;
            // A mirrored block evaluates its derivatives on the conjugate
            // transposes of the bra and of the partially evolved ket, which
            // need two more local blocks of scratch.
            const bool mirrors = runtime_t::kMirrorsHermitianTiles && tile.layout.mirror_shift != 0;
            const auto block_count = 1 + batch_states.size() + (mirrors ? 2 : 0);
            auto* scratch = runtime_t::ThreadScratch(block_count * dimension + tile.layout.workspace_size);
            auto* shared_tile = scratch;
            auto* mirror_bra = runtime_t::Offset(scratch, (1 + batch_states.size()) * dimension);
            auto* mirror_ket = runtime_t::Offset(scratch, (2 + batch_states.size()) * dimension);
            auto* workspace = runtime_t::Offset(scratch, block_count * dimension);
            const auto batch_tile = [&](size_t batch) { return runtime_t::Offset(scratch, (1 + batch) * dimension); };

            for_each(traversals[tile_index], [&](auto active, auto state_base, const auto& index_context) {
                bool mirrored = false;
                auto mirror_base = state_base;
                if constexpr (runtime_t::kMirrorsHermitianTiles) {
                    mirrored = runtime_t::Mirrored(index_context);
                    if (mirrored) {
                        mirror_base = runtime_t::MirrorBase(index_context);
                    }
                }

                runtime_t::Gather(shared_state, tile.layout, active, state_base, index_context, shared_tile);
                for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                    runtime_t::Gather(batch_states[batch], tile.layout, active, state_base, index_context,
                                      batch_tile(batch));
                }

                if (program.compute_overlap && tile_index == 0) {
                    for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                        auto overlap = kernels_t::InnerProduct(active, shared_tile, batch_tile(batch), dimension);
                        if (mirrored) {
                            // The mirror block contributes the conjugate.
                            overlap += std::conj(overlap);
                        }
                        local_output[batch * program.output_size] += conjugate_output ? std::conj(overlap) : overlap;
                    }
                }

                for (const auto& operation : tile.operations) {
                    for (const auto& evolution : operation.evolution) {
                        kernels_t::Apply(evolution, active, state_base, index_context, shared_tile, workspace);
                    }
                    const bool evaluate_mirror = mirrored && !operation.derivatives.empty();
                    if constexpr (runtime_t::kMirrorsHermitianTiles) {
                        if (evaluate_mirror) {
                            runtime_t::MirrorBlock(tile.layout, shared_tile, mirror_bra);
                        }
                    }
                    for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                        auto* ket = batch_tile(batch);
                        if constexpr (runtime_t::kMirrorsHermitianTiles) {
                            // The mirror ket must follow its own evolution:
                            // a partially evolved block is not Hermitian.
                            if (evaluate_mirror) {
                                runtime_t::MirrorBlock(tile.layout, ket, mirror_ket);
                            }
                        }
                        for (size_t step = 0; step < operation.evolution.size(); ++step) {
                            if (step == operation.derivative_position) {
                                for (const auto& derivative : operation.derivatives) {
                                    auto intrinsic = kernels_t::ExpectDerivative(derivative, active, state_base,
                                                                                 index_context, shared_tile, ket);
                                    if constexpr (runtime_t::kMirrorsHermitianTiles) {
                                        if (evaluate_mirror) {
                                            intrinsic += kernels_t::ExpectDerivative(
                                                derivative, active, mirror_base, index_context, mirror_bra, mirror_ket);
                                        }
                                    }
                                    for (const auto& target : derivative.targets) {
                                        auto contribution = intrinsic * target.coefficient;
                                        if (conjugate_output) {
                                            contribution = std::conj(contribution);
                                        }
                                        local_output[batch * program.output_size + target.output_index] += contribution;
                                    }
                                }
                            }
                            kernels_t::Apply(operation.evolution[step], active, state_base, index_context, ket,
                                             workspace);
                            if constexpr (runtime_t::kMirrorsHermitianTiles) {
                                if (evaluate_mirror && step < operation.derivative_position) {
                                    kernels_t::Apply(operation.evolution[step], active, mirror_base, index_context,
                                                     mirror_ket, workspace);
                                }
                            }
                        }
                    }
                }

                runtime_t::Scatter(shared_state, tile.layout, active, state_base, index_context, shared_tile);
                for (size_t batch = 0; batch < batch_states.size(); ++batch) {
                    runtime_t::Scatter(batch_states[batch], tile.layout, active, state_base, index_context,
                                       batch_tile(batch));
                }
            });
        }

        const std::lock_guard<std::mutex> lock(output_mutex);
        for (size_t batch = 0; batch < batch_states.size(); ++batch) {
            for (size_t column = 0; column < program.output_size; ++column) {
                (*output)[batch][column] += local_output[batch * program.output_size + column];
            }
        }
    };
    if (!parallel) {
        sweep([](const auto& traversal, const auto& body) { runtime_t::ForEach(traversal, body); });
        return;
    }
    // A single OpenMP region spans the whole adjoint program.  Each omp-for's
    // implicit barrier orders consecutive tiles without rebuilding the team.
    MQ_DO_PRAGMA(omp parallel) {
        sweep([](const auto& traversal, const auto& body) { runtime_t::ForEachInParallel(traversal, body); });
    }
}
}  // namespace mindquantum::sim::circuit_tiling::detail

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_TILE_ENGINE_H_
