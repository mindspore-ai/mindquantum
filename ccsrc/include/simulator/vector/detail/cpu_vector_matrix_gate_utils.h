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

#ifndef INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_MATRIX_GATE_UTILS_H_
#define INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_MATRIX_GATE_UTILS_H_

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/utils.h"
#include "core/mq_base_types.h"
#include "simulator/state_index.h"

namespace mindquantum::sim::vector::detail::matrix_gate {
inline constexpr size_t kMaxStaticMatrixGateQubits = 4;
inline auto GetStateQubitCount(index_t dim) -> qbit_t {
    if (dim == 0 || (dim & (dim - 1)) != 0) {
        throw std::invalid_argument("State dimension must be a non-zero power of two.");
    }
    qbit_t n_qubits = 0;
    while (dim > 1) {
        dim >>= 1;
        ++n_qubits;
    }
    return n_qubits;
}

inline void ValidateQubitSet(const qbits_t& qubits, qbit_t n_state_qubits, const std::string& name) {
    for (auto qubit : qubits) {
        if (qubit < 0 || qubit >= n_state_qubits) {
            throw std::invalid_argument(name + " qubit " + std::to_string(qubit) + " is out of range.");
        }
    }
    auto sorted = qubits;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument(name + " qubits cannot contain duplicates.");
    }
}

template <typename py_qs_data_t>
void ValidateNQubitsMatrixInputs(const qbits_t& objs, const qbits_t& ctrls,
                                 const std::vector<std::vector<py_qs_data_t>>& gate, index_t dim) {
    const auto n_state_qubits = GetStateQubitCount(dim);
    if (objs.empty()) {
        throw std::invalid_argument("Matrix gate requires at least one object qubit.");
    }
    if (objs.size() > static_cast<size_t>(n_state_qubits)) {
        throw std::invalid_argument("Matrix gate has more object qubits than the state.");
    }
    if (objs.size() >= std::numeric_limits<size_t>::digits / 2) {
        throw std::invalid_argument("Matrix gate is too large for local indexing.");
    }

    ValidateQubitSet(objs, n_state_qubits, "Object");
    ValidateQubitSet(ctrls, n_state_qubits, "Control");

    qbits_t all_qubits;
    all_qubits.reserve(objs.size() + ctrls.size());
    all_qubits.insert(all_qubits.end(), objs.begin(), objs.end());
    all_qubits.insert(all_qubits.end(), ctrls.begin(), ctrls.end());
    std::sort(all_qubits.begin(), all_qubits.end());
    if (std::adjacent_find(all_qubits.begin(), all_qubits.end()) != all_qubits.end()) {
        throw std::invalid_argument("Object qubits and control qubits cannot overlap.");
    }

    const auto gate_dim = static_cast<size_t>(1) << objs.size();
    if (gate.size() != gate_dim) {
        throw std::invalid_argument("Matrix gate row count does not match object qubits.");
    }
    for (const auto& row : gate) {
        if (row.size() != gate_dim) {
            throw std::invalid_argument("Matrix gate rows have inconsistent sizes.");
        }
    }
}

inline auto QubitsToMask(const qbits_t& qubits) -> index_t {
    index_t out = 0;
    for (auto qubit : qubits) {
        out |= static_cast<index_t>(1) << qubit;
    }
    return out;
}

inline auto MakeOccupiedQubits(const qbits_t& objs, const qbits_t& ctrls) -> qbits_t {
    qbits_t occupied;
    occupied.reserve(objs.size() + ctrls.size());
    occupied.insert(occupied.end(), objs.begin(), objs.end());
    occupied.insert(occupied.end(), ctrls.begin(), ctrls.end());
    std::sort(occupied.begin(), occupied.end());
    return occupied;
}

template <typename scalar_t, size_t static_size>
using StaticOrDynamicBuffer = std::conditional_t<static_size == 0, std::vector<scalar_t>,
                                                 std::array<scalar_t, static_size>>;

template <typename scalar_t, size_t static_size>
auto MakeBuffer(size_t runtime_size) -> StaticOrDynamicBuffer<scalar_t, static_size> {
    if constexpr (static_size == 0) {
        return StaticOrDynamicBuffer<scalar_t, static_size>(runtime_size);
    } else {
        (void) runtime_size;
        return {};
    }
}

inline void FillObjectMasks(const qbits_t& objs, size_t gate_dim, index_t* obj_masks) {
    std::fill_n(obj_masks, gate_dim, index_t{0});
    for (size_t basis = 0; basis < gate_dim; ++basis) {
        for (size_t bit = 0; bit < objs.size(); ++bit) {
            if ((basis & (static_cast<size_t>(1) << bit)) != 0) {
                obj_masks[basis] |= static_cast<index_t>(1) << objs[bit];
            }
        }
    }
}

template <size_t static_gate_dim = 0>
auto MakeObjectMasks(const qbits_t& objs, size_t gate_dim) -> StaticOrDynamicBuffer<index_t, static_gate_dim> {
    auto obj_masks = MakeBuffer<index_t, static_gate_dim>(gate_dim);
    FillObjectMasks(objs, gate_dim, obj_masks.data());
    return obj_masks;
}

template <typename qs_data_t, typename py_qs_data_t>
void FlattenGateTo(const std::vector<std::vector<py_qs_data_t>>& gate, size_t gate_dim, qs_data_t* gate_flat) {
    for (size_t row = 0; row < gate_dim; ++row) {
        for (size_t col = 0; col < gate_dim; ++col) {
            gate_flat[row * gate_dim + col] = gate[row][col];
        }
    }
}

template <size_t static_gate_dim = 0, typename qs_data_t, typename py_qs_data_t>
auto FlattenGate(const std::vector<std::vector<py_qs_data_t>>& gate, size_t gate_dim)
    -> StaticOrDynamicBuffer<qs_data_t, static_gate_dim == 0 ? 0 : static_gate_dim * static_gate_dim> {
    auto gate_flat = MakeBuffer<qs_data_t, static_gate_dim == 0 ? 0 : static_gate_dim * static_gate_dim>(
        gate_dim * gate_dim);
    FlattenGateTo(gate, gate_dim, gate_flat.data());
    return gate_flat;
}

template <size_t static_gate_dim, typename qs_data_t>
struct PreparedMatrixGate {
    using mask_buffer_t = StaticOrDynamicBuffer<index_t, static_gate_dim>;
    using gate_buffer_t
        = StaticOrDynamicBuffer<qs_data_t, static_gate_dim == 0 ? 0 : static_gate_dim * static_gate_dim>;

    PreparedMatrixGate(size_t gate_dim, mask_buffer_t obj_masks, qbits_t occupied_qubits, index_t ctrl_mask,
                       index_t block_count, gate_buffer_t gate_flat)
        : gate_dim(gate_dim),
          obj_masks(std::move(obj_masks)),
          occupied_qubits(std::move(occupied_qubits)),
          bit_expander(this->occupied_qubits),
          ctrl_mask(ctrl_mask),
          block_count(block_count),
          gate_flat(std::move(gate_flat)) {
    }

    size_t gate_dim;
    mask_buffer_t obj_masks;
    qbits_t occupied_qubits;
    state_index::BitExpander bit_expander;
    index_t ctrl_mask;
    index_t block_count;
    gate_buffer_t gate_flat;
};

template <size_t static_gate_dim, typename qs_data_t, typename py_qs_data_t>
auto PrepareMatrixGate(const qbits_t& objs, const qbits_t& ctrls,
                       const std::vector<std::vector<py_qs_data_t>>& gate, index_t dim)
    -> PreparedMatrixGate<static_gate_dim, qs_data_t> {
    const auto gate_dim = static_gate_dim == 0 ? static_cast<size_t>(1) << objs.size() : static_gate_dim;
    auto obj_masks = MakeObjectMasks<static_gate_dim>(objs, gate_dim);
    auto occupied_qubits = MakeOccupiedQubits(objs, ctrls);
    const auto ctrl_mask = QubitsToMask(ctrls);
    const auto block_count = dim >> occupied_qubits.size();
    auto gate_flat = FlattenGate<static_gate_dim, qs_data_t>(gate, gate_dim);
    return {gate_dim, std::move(obj_masks), std::move(occupied_qubits), ctrl_mask, block_count,
            std::move(gate_flat)};
}

template <size_t static_gate_dim, typename qs_data_t, typename block_func_t>
void ForEachMatrixGateBlock(index_t dim, index_t block_count, size_t gate_dim, const block_func_t& block_func) {
    if constexpr (static_gate_dim == 0) {
        if (const int team = omp::TeamSize(dim); team == 1) {
            std::vector<qs_data_t> input(gate_dim);
            for (index_t block = 0; block < block_count; ++block) {
                block_func(block, input.data());
            }
        } else {
            MQ_DO_PRAGMA(omp parallel num_threads(team)) {
                std::vector<qs_data_t> input(gate_dim);
                MQ_DO_PRAGMA(omp for schedule(static))
                for (omp::idx_t block = 0; block < static_cast<omp::idx_t>(block_count); ++block) {
                    block_func(static_cast<index_t>(block), input.data());
                }
            }
        }
    } else {
        MQ_OMP_PARALLEL_FOR(
            dim, for (omp::idx_t block = 0; block < static_cast<omp::idx_t>(block_count); ++block) {
                std::array<qs_data_t, static_gate_dim> input;
                block_func(static_cast<index_t>(block), input.data());
            })
    }
}

template <size_t static_gate_dim, typename qs_data_t, typename calc_type, typename block_func_t>
auto ReduceMatrixGateBlocks(index_t dim, index_t block_count, size_t gate_dim,
                            const block_func_t& block_func) -> qs_data_t {
    calc_type res_real = 0;
    calc_type res_imag = 0;
    if constexpr (static_gate_dim == 0) {
        if (const int team = omp::TeamSize(dim); team == 1) {
            std::vector<qs_data_t> input(gate_dim);
            for (index_t block = 0; block < block_count; ++block) {
                const auto block_sum = block_func(block, input.data());
                res_real += block_sum.real();
                res_imag += block_sum.imag();
            }
        } else {
            MQ_DO_PRAGMA(omp parallel num_threads(team) reduction(+ : res_real, res_imag)) {
                std::vector<qs_data_t> input(gate_dim);
                MQ_DO_PRAGMA(omp for schedule(static))
                for (omp::idx_t block = 0; block < static_cast<omp::idx_t>(block_count); ++block) {
                    const auto block_sum = block_func(static_cast<index_t>(block), input.data());
                    res_real += block_sum.real();
                    res_imag += block_sum.imag();
                }
            }
        }
    } else {
        MQ_OMP_PARALLEL_FOR_CLAUSES(
            dim, reduction(+ : res_real, res_imag) schedule(static),
            for (omp::idx_t block = 0; block < static_cast<omp::idx_t>(block_count); ++block) {
                std::array<qs_data_t, static_gate_dim> input;
                const auto block_sum = block_func(static_cast<index_t>(block), input.data());
                res_real += block_sum.real();
                res_imag += block_sum.imag();
            })
    }
    return {res_real, res_imag};
}

template <size_t current_n_qubits, size_t max_n_qubits, typename kernel_t>
auto DispatchStaticNQubitsImpl(size_t n_qubits, kernel_t& kernel)
    -> decltype(kernel(std::integral_constant<size_t, 0>{})) {
    if constexpr (current_n_qubits > max_n_qubits) {
        return kernel(std::integral_constant<size_t, 0>{});
    } else {
        if (n_qubits == current_n_qubits) {
            return kernel(std::integral_constant<size_t, static_cast<size_t>(1) << current_n_qubits>{});
        }
        return DispatchStaticNQubitsImpl<current_n_qubits + 1, max_n_qubits>(n_qubits, kernel);
    }
}

template <typename kernel_t>
auto DispatchStaticNQubits(size_t n_qubits, kernel_t&& kernel)
    -> decltype(kernel(std::integral_constant<size_t, 0>{})) {
    return DispatchStaticNQubitsImpl<3, kMaxStaticMatrixGateQubits>(n_qubits, kernel);
}
}  // namespace mindquantum::sim::vector::detail::matrix_gate

#endif  // INCLUDE_SIMULATOR_VECTOR_DETAIL_CPU_VECTOR_MATRIX_GATE_UTILS_H_
