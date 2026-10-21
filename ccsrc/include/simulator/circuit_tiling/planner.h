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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_PLANNER_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_PLANNER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/circuit_tiling/state_space.h"

namespace mindquantum::sim::circuit_tiling {
// Groups a circuit into cache-resident tiles of state-index bits.  The
// planner reasons purely about state accesses, so a density matrix circuit
// is planned exactly like a state vector circuit on twice the index bits.
class Planner {
 public:
    // Vector execution may leave controls/diagonals outside the tile. A fixed
    // dense operator instead needs its entire support inside the capacity.
    enum class Capacity { Mixing, FullSupport };

 private:
    // A calibrated, bounded frontier prevents locally attractive masks from
    // fragmenting later tiles.  One ordering keeps the policy deterministic
    // and explainable; twelve candidates retain the measured plan quality
    // without the former dual-heuristic search.
    static constexpr size_t kCandidateWidth = 12;

    using segment_t = std::vector<GateAccess>;

    struct DependencyGraph {
        segment_t gates;
        std::vector<std::vector<size_t>> successors;
        std::vector<size_t> indegree;
    };

    struct TileEvaluation {
        index_t qubit_mask = 0;
        std::vector<size_t> executed;
        std::vector<size_t> frontier;
    };

 public:
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;

    explicit Planner(const StateSpace& space, size_t max_tile_qubits = kMaxTileQubits,
                     Capacity capacity = Capacity::Mixing)
        : space_(space), n_qubits_(space.state_qubits()), max_tile_qubits_(max_tile_qubits), capacity_(capacity) {
        if (max_tile_qubits_ == 0 || max_tile_qubits_ >= std::numeric_limits<index_t>::digits) {
            throw std::invalid_argument("Circuit tile qubit count is outside the local index range.");
        }
    }

    [[nodiscard]] auto Plan(const circuit_t& circuit) const -> Schedule {
        return PlanImpl(circuit, false);
    }

    [[nodiscard]] auto PlanAdjoint(const circuit_t& circuit) const -> Schedule {
        // A single gate still participates in the adjoint tile program: its
        // derivative and the bra/ket evolution share one state gather.
        return PlanImpl(circuit, true);
    }

 private:
    StateSpace space_;
    qbit_t n_qubits_;
    size_t max_tile_qubits_;
    Capacity capacity_;

    [[nodiscard]] static auto RequiredMask(const GateAccess& access, Capacity capacity) noexcept -> index_t {
        return capacity == Capacity::FullSupport ? access.object_mask | access.control_mask : access.mixing_mask;
    }

    [[nodiscard]] auto PlanImpl(const circuit_t& circuit, bool keep_single_gate_tiles) const -> Schedule {
        Schedule schedule;
        schedule.reserve(circuit.size());
        segment_t segment;
        segment.reserve(circuit.size());

        for (const auto& gate : circuit) {
            if (gate == nullptr) {
                throw std::invalid_argument("Cannot plan a null gate.");
            }
            if (gate->GetID() == GateID::I) {
                continue;
            }
            if (auto access = DescribeGateAccess(gate, space_);
                access.has_value() && PopCount(RequiredMask(*access, capacity_)) <= max_tile_qubits_) {
                segment.push_back(std::move(*access));
                continue;
            }
            FlushSegment(&schedule, &segment, keep_single_gate_tiles);
            schedule.emplace_back(gate);
        }
        FlushSegment(&schedule, &segment, keep_single_gate_tiles);
        return schedule;
    }

    [[nodiscard]] static auto PopCount(index_t value) noexcept -> size_t {
        return static_cast<size_t>(CountOne(static_cast<std::uint64_t>(value)));
    }

    template <typename function_t>
    void ForEachQubit(index_t mask, function_t&& function) const {
        for (qbit_t qubit = 0; qubit < n_qubits_; ++qubit) {
            if ((mask & (index_t{1} << qubit)) != 0) {
                function(qubit);
            }
        }
    }

    // Diagonal operands and controls become tile-local while the tile has
    // room, most frequently accessed first.  Every image of a circuit qubit
    // enters together, so density matrix tiles stay image-symmetric and the
    // engine can evolve only half of their blocks.
    [[nodiscard]] auto SelectTileQubits(index_t mixing_mask, const std::vector<size_t>& gate_indices,
                                        const DependencyGraph& graph) const -> qbits_t {
        const auto n_circuit_qubits = space_.circuit_qubits();
        if (n_circuit_qubits == 0) {
            return QubitsFromMask(mixing_mask);
        }
        const auto images = static_cast<size_t>(n_qubits_ / n_circuit_qubits);
        std::vector<size_t> access_weights(static_cast<size_t>(n_circuit_qubits), 0);
        for (auto gate_index : gate_indices) {
            const auto non_mixing_access = (graph.gates[gate_index].object_mask | graph.gates[gate_index].control_mask)
                                           & ~mixing_mask;
            ForEachQubit(non_mixing_access, [&access_weights, n_circuit_qubits](qbit_t qubit) {
                ++access_weights[static_cast<size_t>(qubit % n_circuit_qubits)];
            });
        }

        std::vector<qbit_t> candidates;
        for (qbit_t qubit = 0; qubit < n_circuit_qubits; ++qubit) {
            if (access_weights[static_cast<size_t>(qubit)] != 0) {
                candidates.push_back(qubit);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&access_weights](qbit_t lhs, qbit_t rhs) {
            const auto lhs_weight = access_weights[static_cast<size_t>(lhs)];
            const auto rhs_weight = access_weights[static_cast<size_t>(rhs)];
            return lhs_weight != rhs_weight ? lhs_weight > rhs_weight : lhs < rhs;
        });

        auto tile_mask = mixing_mask;
        auto tile_size = PopCount(tile_mask);
        for (auto qubit : candidates) {
            if (tile_size + images > max_tile_qubits_) {
                break;
            }
            const auto image_mask = space_.Embed(index_t{1} << qubit);
            if ((tile_mask & image_mask) == 0) {
                tile_mask |= image_mask;
                tile_size += images;
            }
        }
        return QubitsFromMask(tile_mask);
    }

    [[nodiscard]] auto BuildDependencyGraph(segment_t segment) const -> DependencyGraph {
        DependencyGraph graph;
        graph.gates = std::move(segment);
        graph.successors.resize(graph.gates.size());
        graph.indegree.resize(graph.gates.size(), 0);

        // Mixing accesses conflict with every access on a qubit.  Diagonal
        // operands and controls commute with one another, so they only depend
        // on the previous mixer.  This is the only reordering rule.
        std::vector<std::optional<size_t>> last_mixer(static_cast<size_t>(n_qubits_));
        std::vector<std::vector<size_t>> non_mixing_users(static_cast<size_t>(n_qubits_));
        for (size_t gate_index = 0; gate_index < graph.gates.size(); ++gate_index) {
            const auto& gate = graph.gates[gate_index];
            std::vector<size_t> predecessors;
            ForEachQubit(gate.control_mask, [&](qbit_t control) {
                if (const auto mixer = last_mixer[static_cast<size_t>(control)]; mixer.has_value()) {
                    predecessors.push_back(*mixer);
                }
            });
            ForEachQubit(gate.object_mask, [&](qbit_t object) {
                const auto object_index = static_cast<size_t>(object);
                if (const auto mixer = last_mixer[object_index]; mixer.has_value()) {
                    predecessors.push_back(*mixer);
                }
                if ((gate.mixing_mask & (index_t{1} << object)) != 0) {
                    predecessors.insert(predecessors.end(), non_mixing_users[object_index].begin(),
                                        non_mixing_users[object_index].end());
                }
            });
            std::sort(predecessors.begin(), predecessors.end());
            predecessors.erase(std::unique(predecessors.begin(), predecessors.end()), predecessors.end());
            for (auto predecessor : predecessors) {
                graph.successors[predecessor].push_back(gate_index);
                ++graph.indegree[gate_index];
            }

            ForEachQubit(gate.object_mask, [&](qbit_t object) {
                const auto object_index = static_cast<size_t>(object);
                if ((gate.mixing_mask & (index_t{1} << object)) != 0) {
                    non_mixing_users[object_index].clear();
                    last_mixer[object_index] = gate_index;
                } else {
                    non_mixing_users[object_index].push_back(gate_index);
                }
            });
            ForEachQubit(gate.control_mask,
                         [&](qbit_t control) { non_mixing_users[static_cast<size_t>(control)].push_back(gate_index); });
        }
        return graph;
    }

    // Computes the exact topological closure executable inside one candidate
    // tile.  Reusable working storage and an undo log keep candidate evaluation
    // linear in the touched subgraph without obscuring the scheduling policy.
    class ClosureEvaluator {
     public:
        ClosureEvaluator(const DependencyGraph& graph, std::vector<size_t>& indegree, const std::set<size_t>& ready,
                         Capacity capacity)
            : graph_(graph)
            , ready_(ready)
            , capacity_(capacity)
            , working_indegree_(indegree)
            , locally_executed_(graph.gates.size(), 0) {
            assert(graph_.gates.size() == graph_.successors.size());
            assert(graph_.gates.size() == indegree.size());
            assert(graph_.gates.empty() || !ready_.empty());
            executable_.reserve(ready_.size());
            newly_ready_.reserve(ready_.size());
            decremented_successors_.reserve(graph_.gates.size());
        }

        [[nodiscard]] auto Evaluate(index_t qubit_mask) -> TileEvaluation {
            executable_.clear();
            newly_ready_.clear();
            decremented_successors_.clear();
            for (auto gate_index : ready_) {
                assert(gate_index < graph_.gates.size() && working_indegree_[gate_index] == 0);
                if (IsCompatible(graph_.gates[gate_index], qubit_mask)) {
                    executable_.push_back(gate_index);
                }
            }
            std::make_heap(executable_.begin(), executable_.end(), std::greater<size_t>{});

            TileEvaluation evaluation;
            evaluation.qubit_mask = qubit_mask;
            while (!executable_.empty()) {
                std::pop_heap(executable_.begin(), executable_.end(), std::greater<size_t>{});
                const auto gate_index = executable_.back();
                executable_.pop_back();
                locally_executed_[gate_index] = 1;
                evaluation.executed.push_back(gate_index);

                for (auto successor : graph_.successors[gate_index]) {
                    auto& degree = working_indegree_[successor];
                    assert(degree != 0);
                    --degree;
                    decremented_successors_.push_back(successor);
                    if (degree == 0) {
                        newly_ready_.push_back(successor);
                        if (IsCompatible(graph_.gates[successor], qubit_mask)) {
                            executable_.push_back(successor);
                            std::push_heap(executable_.begin(), executable_.end(), std::greater<size_t>{});
                        }
                    }
                }
            }

            for (auto gate_index : ready_) {
                if (locally_executed_[gate_index] == 0) {
                    evaluation.frontier.push_back(gate_index);
                }
            }
            for (auto gate_index : newly_ready_) {
                if (locally_executed_[gate_index] == 0) {
                    evaluation.frontier.push_back(gate_index);
                }
            }
            std::sort(evaluation.frontier.begin(), evaluation.frontier.end());
            Rollback(evaluation.executed);
            return evaluation;
        }

     private:
        const DependencyGraph& graph_;
        const std::set<size_t>& ready_;
        Capacity capacity_;
        // Every candidate rolls back its edits before the scheduler commits a
        // tile. Share that live degree array and allocate scratch once per
        // segment, not once per tile (which costs O(gates * tiles)).
        std::vector<size_t>& working_indegree_;
        std::vector<uint8_t> locally_executed_;
        std::vector<size_t> executable_;
        std::vector<size_t> newly_ready_;
        std::vector<size_t> decremented_successors_;

        [[nodiscard]] auto IsCompatible(const GateAccess& gate, index_t qubit_mask) const -> bool {
            return (RequiredMask(gate, capacity_) & ~qubit_mask) == 0;
        }

        void Rollback(const std::vector<size_t>& executed) {
            // Restore the degrees seen on entry, not the original graph's
            // degrees: earlier tiles have already committed. Keep repeated
            // successors in the log to undo every edge of a fan-in.
            for (auto successor : decremented_successors_) {
                ++working_indegree_[successor];
            }
            for (auto gate_index : executed) {
                locally_executed_[gate_index] = 0;
            }
        }
    };

    [[nodiscard]] static auto BetterTile(const TileEvaluation& lhs, const TileEvaluation& rhs) -> bool {
        if (lhs.executed.size() != rhs.executed.size()) {
            return lhs.executed.size() > rhs.executed.size();
        }
        const auto lhs_qubits = PopCount(lhs.qubit_mask);
        const auto rhs_qubits = PopCount(rhs.qubit_mask);
        if (lhs_qubits != rhs_qubits) {
            return lhs_qubits < rhs_qubits;
        }
        const auto lhs_first = lhs.executed.empty() ? std::numeric_limits<size_t>::max() : lhs.executed.front();
        const auto rhs_first = rhs.executed.empty() ? std::numeric_limits<size_t>::max() : rhs.executed.front();
        if (lhs_first != rhs_first) {
            return lhs_first < rhs_first;
        }
        if (lhs.frontier.size() != rhs.frontier.size()) {
            return lhs.frontier.size() < rhs.frontier.size();
        }
        return lhs.qubit_mask < rhs.qubit_mask;
    }

    [[nodiscard]] auto ChooseTile(const DependencyGraph& graph, const std::set<size_t>& ready,
                                  ClosureEvaluator& evaluator) const -> TileEvaluation {
        const auto expected_candidates = ready.size() + kCandidateWidth * max_tile_qubits_;
        std::vector<TileEvaluation> evaluated;
        evaluated.reserve(expected_candidates);
        std::unordered_set<index_t> evaluated_masks;
        evaluated_masks.reserve(expected_candidates);
        const auto evaluate = [&evaluated, &evaluated_masks, &evaluator](index_t mask) {
            if (evaluated_masks.insert(mask).second) {
                evaluated.push_back(evaluator.Evaluate(mask));
            }
        };

        std::vector<index_t> seed_masks;
        seed_masks.reserve(ready.size());
        std::transform(ready.begin(), ready.end(), std::back_inserter(seed_masks),
                       [&](auto gate_index) { return RequiredMask(graph.gates[gate_index], capacity_); });
        std::sort(seed_masks.begin(), seed_masks.end());
        seed_masks.erase(std::unique(seed_masks.begin(), seed_masks.end()), seed_masks.end());
        for (auto mask : seed_masks) {
            evaluate(mask);
        }
        assert(!evaluated.empty());

        const auto select_candidates = [&evaluated]() {
            std::vector<size_t> ranked(evaluated.size());
            std::iota(ranked.begin(), ranked.end(), 0);
            const auto count = std::min(kCandidateWidth, ranked.size());
            std::partial_sort(
                ranked.begin(), ranked.begin() + count, ranked.end(),
                [&evaluated](size_t lhs, size_t rhs) { return BetterTile(evaluated[lhs], evaluated[rhs]); });
            ranked.resize(count);
            return ranked;
        };

        auto candidates = select_candidates();
        for (size_t round = 0; round < max_tile_qubits_; ++round) {
            std::vector<index_t> expansions;
            for (auto candidate_index : candidates) {
                const auto& candidate = evaluated[candidate_index];
                for (auto gate_index : candidate.frontier) {
                    const auto expanded = candidate.qubit_mask | RequiredMask(graph.gates[gate_index], capacity_);
                    if (expanded != candidate.qubit_mask && PopCount(expanded) <= max_tile_qubits_) {
                        expansions.push_back(expanded);
                    }
                }
            }
            std::sort(expansions.begin(), expansions.end());
            expansions.erase(std::unique(expansions.begin(), expansions.end()), expansions.end());
            if (expansions.empty()) {
                break;
            }

            const auto previous_count = evaluated.size();
            for (auto mask : expansions) {
                evaluate(mask);
            }
            if (evaluated.size() == previous_count) {
                break;
            }
            candidates = select_candidates();
        }

        const auto best = std::min_element(evaluated.begin(), evaluated.end(),
                                           [](const auto& lhs, const auto& rhs) { return BetterTile(lhs, rhs); });
        assert(best != evaluated.end() && !best->executed.empty());
        return *best;
    }

    [[nodiscard]] auto ScheduleSegment(segment_t segment, bool keep_single_gate_tiles) const -> Schedule {
        auto graph = BuildDependencyGraph(std::move(segment));
        auto indegree = graph.indegree;
        std::vector<bool> scheduled(graph.gates.size(), false);
        std::set<size_t> ready;
        for (size_t gate_index = 0; gate_index < graph.gates.size(); ++gate_index) {
            if (indegree[gate_index] == 0) {
                ready.insert(gate_index);
            }
        }

        Schedule schedule;
        schedule.reserve(graph.gates.size());
        ClosureEvaluator evaluator(graph, indegree, ready, capacity_);
        size_t remaining = graph.gates.size();
        while (remaining != 0) {
            const auto selected = ChooseTile(graph, ready, evaluator);
            CircuitTile tile;
            index_t mixing_mask = 0;
            tile.gates.reserve(selected.executed.size());
            for (auto gate_index : selected.executed) {
                assert(!scheduled[gate_index] && indegree[gate_index] == 0);
                scheduled[gate_index] = true;
                --remaining;
                ready.erase(gate_index);
                mixing_mask |= graph.gates[gate_index].mixing_mask;
                tile.gates.push_back(graph.gates[gate_index].gate);
                for (auto successor : graph.successors[gate_index]) {
                    assert(indegree[successor] != 0);
                    if (--indegree[successor] == 0) {
                        ready.insert(successor);
                    }
                }
            }
            tile.qubits = SelectTileQubits(mixing_mask, selected.executed, graph);
            if (tile.gates.size() == 1 && !keep_single_gate_tiles) {
                schedule.emplace_back(tile.gates.front());
            } else {
                schedule.emplace_back(std::move(tile));
            }
        }
        return schedule;
    }

    void FlushSegment(Schedule* schedule, segment_t* segment, bool keep_single_gate_tiles) const {
        if (segment->empty()) {
            return;
        }
        auto scheduled = ScheduleSegment(std::move(*segment), keep_single_gate_tiles);
        schedule->insert(schedule->end(), std::make_move_iterator(scheduled.begin()),
                         std::make_move_iterator(scheduled.end()));
        segment->clear();
    }
};
}  // namespace mindquantum::sim::circuit_tiling

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_PLANNER_H_
