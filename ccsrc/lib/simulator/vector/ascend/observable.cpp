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
#include "simulator/vector/ascend/observable.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "ops/hamiltonian.h"
#include "qubit_layout.h"

namespace mindquantum::sim::ascend {
namespace {
uint64_t Dimension(int64_t n_qubits) {
    if (n_qubits < 0 || n_qubits > 32) {
        throw std::invalid_argument("Ascend observables require 0..32 qubits.");
    }
    return uint64_t{1} << n_qubits;
}

void ValidateCoefficient(float value) {
    // Bit classification remains correct even if a consuming host target uses
    // -ffast-math. Do not let finite-math-only erase validation of user input.
    static_assert(sizeof(float) == sizeof(uint32_t), "float32 required");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        throw std::invalid_argument("Ascend observable coefficients must be finite float32 values.");
    }
}

uint32_t Parity(uint64_t value) {
    value ^= value >> 32;
    value ^= value >> 16;
    value ^= value >> 8;
    value ^= value >> 4;
    return (0x6996U >> (value & 15U)) & 1U;
}
}  // namespace

ObservableProgram::ObservableProgram(ObservableLayout layout, std::vector<ObservableGroup> groups,
                                     std::vector<ObservableTerm> terms, std::vector<uint32_t> gather_indices,
                                     std::vector<float> signs, std::vector<uint64_t> row_offsets,
                                     std::vector<uint64_t> columns, std::vector<ObservableValue> values)
    : layout_(layout)
    , groups_(std::move(groups))
    , terms_(std::move(terms))
    , gather_indices_(std::move(gather_indices))
    , signs_(std::move(signs))
    , row_offsets_(std::move(row_offsets))
    , columns_(std::move(columns))
    , values_(std::move(values)) {
}

ObservableProgram ObservableProgram::Compile(const Hamiltonian<float>& hamiltonian, int64_t n_qubits) {
    const auto dimension = Dimension(n_qubits);
    const auto plane = std::max(uint64_t{8}, dimension);
    const auto tile = static_cast<uint32_t>(std::min(plane, uint64_t{kObservableTileSize}));
    ObservableLayout layout{dimension, plane, 0, kObservablePauli, tile};
    if (hamiltonian.how_to_ != ORIGIN && hamiltonian.how_to_ != BACKEND && hamiltonian.how_to_ != FRONTEND) {
        throw std::invalid_argument("Unknown Ascend Hamiltonian representation.");
    }
    if (hamiltonian.how_to_ != ORIGIN && hamiltonian.n_qubits_ != static_cast<Index>(n_qubits)) {
        throw std::invalid_argument("Ascend Hamiltonian qubit count does not match the state.");
    }
    return hamiltonian.how_to_ == FRONTEND ? CompileCsr(hamiltonian, layout) : CompilePauli(hamiltonian, layout);
}

ObservableProgram ObservableProgram::CompileCsr(const Hamiltonian<float>& hamiltonian, ObservableLayout layout) {
    const auto dimension = layout.dimension;
    const auto& csr = hamiltonian.ham_sparse_main_;
    if (!csr || csr->dim_ != dimension || !csr->indptr_ || (csr->nnz_ && (!csr->indices_ || !csr->data_))) {
        throw std::invalid_argument("Invalid Ascend observable CSR shape or missing storage.");
    }
    if (csr->indptr_[0] != 0 || csr->indptr_[dimension] != csr->nnz_) {
        throw std::invalid_argument("Ascend observable CSR row endpoints must be 0 and nnz.");
    }
    // Validate before copying, including zero coefficients and empty rows.
    // Unsorted/duplicate columns are valid and remain separate entries.
    for (uint64_t row = 0; row < dimension; ++row) {
        if (csr->indptr_[row] > csr->indptr_[row + 1] || csr->indptr_[row + 1] > csr->nnz_) {
            throw std::invalid_argument("Ascend observable CSR row offsets must be monotone and <= nnz.");
        }
    }
    for (uint64_t i = 0; i < csr->nnz_; ++i) {
        if (csr->indices_[i] >= dimension) {
            throw std::invalid_argument("Ascend observable CSR column is outside the state.");
        }
        ValidateCoefficient(csr->data_[i].real());
        ValidateCoefficient(csr->data_[i].imag());
    }
    std::vector<uint64_t> rows(csr->indptr_, csr->indptr_ + dimension + 1);
    std::vector<uint64_t> columns;
    std::vector<ObservableValue> values;
    columns.reserve(csr->nnz_);
    values.reserve(csr->nnz_);
    for (uint64_t i = 0; i < csr->nnz_; ++i) {
        columns.push_back(csr->indices_[i]);
        values.push_back({csr->data_[i].real(), csr->data_[i].imag()});
    }
    layout.kind = kObservableCsr;
    return ObservableProgram(layout, {}, {}, {}, {}, std::move(rows), std::move(columns), std::move(values));
}

namespace {
std::map<uint64_t, std::vector<ObservableTerm>> GroupPauliTerms(const Hamiltonian<float>& hamiltonian,
                                                                uint64_t dimension) {
    std::map<uint64_t, std::vector<ObservableTerm>> grouped;
    for (const auto& [word, coefficient] : hamiltonian.ham_) {
        ValidateCoefficient(coefficient);
        uint64_t seen = 0, flip = 0, parity = 0;
        uint32_t ys = 0;
        for (const auto& [qubit, axis] : word) {
            if (qubit >= 32 || (uint64_t{1} << qubit) >= dimension) {
                throw std::invalid_argument("Ascend observable Pauli qubit is outside the state.");
            }
            const auto bit = uint64_t{1} << qubit;
            if (seen & bit) {
                throw std::invalid_argument("Ascend observable Pauli words must have distinct qubits.");
            }
            seen |= bit;
            if (axis == 'X' || axis == 'Y') {
                flip |= bit;
            }
            if (axis == 'Y' || axis == 'Z') {
                parity |= bit;
            }
            if (axis == 'Y') {
                ++ys;
            } else if (axis != 'I' && axis != 'X' && axis != 'Z') {
                throw std::invalid_argument("Ascend observable Pauli axes must be I, X, Y or Z.");
            }
        }
        // In output basis j, P|ket>[j] = (-i)^ny * (-1)^popcount(j & (Y|Z))
        // * ket[j ^ (X|Y)]. The source-basis i^ny formula has the opposite Y
        // sign, a distinction important for asymmetric complex ket/bra pairs.
        ObservableTerm term{};
        term.parity_mask = parity;
        switch (ys & 3U) {
            case 0:
                term.coefficient.real = coefficient;
                break;
            case 1:
                term.coefficient.imag = -coefficient;
                break;
            case 2:
                term.coefficient.real = -coefficient;
                break;
            default:
                term.coefficient.imag = coefficient;
                break;
        }
        // Validate even zero terms, but do not upload or execute them.
        if (coefficient != 0.0F) {
            grouped[flip].push_back(term);
        }
    }
    return grouped;
}
}  // namespace

ObservableProgram ObservableProgram::CompilePauli(const Hamiltonian<float>& hamiltonian, ObservableLayout layout) {
    const auto tile = layout.tile_size;
    std::vector<ObservableGroup> groups;
    std::vector<ObservableTerm> terms;
    std::vector<uint32_t> indices;
    std::vector<float> signs;
    std::map<uint32_t, uint32_t> index_offsets, sign_offsets;
    // Pools contain at most tile distinct patterns of tile elements (<=4 MiB
    // each), irrespective of qubit count or number of terms. Offsets fit u32.
    for (const auto& [flip, group_terms] : GroupPauliTerms(hamiltonian, layout.dimension)) {
        const auto low_flip = static_cast<uint32_t>(flip & (tile - 1));
        auto [index_it, new_index] = index_offsets.emplace(low_flip, static_cast<uint32_t>(indices.size()));
        if (new_index) {
            for (uint32_t i = 0; i < tile; ++i) {
                indices.push_back((i ^ low_flip) * sizeof(float));  // Gather uses BYTE offsets
            }
        }
        groups.push_back({flip, terms.size(), group_terms.size(), index_it->second, 0});
        for (auto term : group_terms) {
            const auto low_parity = static_cast<uint32_t>(term.parity_mask & (tile - 1));
            auto [sign_it, new_sign] = sign_offsets.emplace(low_parity, static_cast<uint32_t>(signs.size()));
            if (new_sign) {
                for (uint32_t i = 0; i < tile; ++i) {
                    signs.push_back(Parity(i & low_parity) ? -1.0F : 1.0F);
                }
            }
            term.sign_offset = sign_it->second;
            terms.push_back(term);
        }
    }
    layout.group_count = groups.size();
    return ObservableProgram(layout, std::move(groups), std::move(terms), std::move(indices), std::move(signs), {}, {},
                             {});
}

ObservableProgram ObservableProgram::Dagger() const {
    if (layout_.kind == kObservablePauli) {
        return *this;
    }
    std::vector<uint64_t> rows(row_offsets_.size(), 0), columns(columns_.size());
    std::vector<ObservableValue> values(values_.size());
    for (const auto column : columns_) {
        ++rows[column + 1];
    }
    std::partial_sum(rows.begin(), rows.end(), rows.begin());
    auto next = rows;
    for (uint64_t row = 0; row < layout_.dimension; ++row) {
        for (auto i = row_offsets_[row]; i < row_offsets_[row + 1]; ++i) {
            const auto target = next[columns_[i]]++;
            columns[target] = row;
            values[target] = {values_[i].real, -values_[i].imag};
        }
    }
    return ObservableProgram(layout_, {}, {}, {}, {}, std::move(rows), std::move(columns), std::move(values));
}

SamplingLayout MakeSamplingLayout(int64_t n_qubits, uint32_t bins) {
    const auto dimension = Dimension(n_qubits);
    if (!bins || bins > kObservableMaxBlocks) {
        throw std::invalid_argument("Ascend sampling requires 1..48 bins.");
    }
    const auto plane = std::max(uint64_t{8}, dimension);
    const auto tiles = (plane + kObservableTileSize - 1) / kObservableTileSize;
    const auto required = (tiles + bins - 1) / bins;
    uint32_t leaves = 8;  // every bin, including unused bins, has a 64-byte-aligned tree
    while (leaves < required) {
        leaves <<= 1;
    }
    return {dimension, plane, bins, leaves, 0};
}

DensityProgram CompileDensity(int64_t n_qubits, uint64_t kept_mask, uint32_t cores) {
    const auto dimension = Dimension(n_qubits);
    if ((kept_mask & ~(dimension - 1)) || !cores || cores > kObservableMaxBlocks) {
        throw std::invalid_argument("Invalid Ascend reduced-density qubits or core count.");
    }
    const auto environment = (dimension - 1) & ~kept_mask;
    uint64_t packet = 7;
    for (uint32_t bit = 3; bit < n_qubits && CountOne(packet) < 10; ++bit) {
        packet |= environment & (uint64_t{1} << bit);
    }
    const auto output_dimension = uint64_t{1} << CountOne(kept_mask);
    if (output_dimension > std::numeric_limits<size_t>::max() / output_dimension / sizeof(ReductionPartial)) {
        throw std::length_error("Reduced density matrix exceeds the allocation address space.");
    }
    const auto pairs = output_dimension * output_dimension;
    const auto tile_count = uint64_t{1} << CountOne(environment & ~packet);
    DensityProgram result{};
    result.layout = {dimension,
                     kept_mask,
                     environment & ~packet,
                     tile_count,
                     output_dimension,
                     uint32_t{1} << CountOne(packet),
                     1,
                     uint32_t{1} << CountOne(environment & packet),
                     static_cast<uint32_t>(std::min(tile_count, (cores + pairs - 1) / pairs))};
    while (packet & result.layout.run) {
        result.layout.run <<= 1;
    }
    result.offsets = Offsets(packet & ~(static_cast<uint64_t>(result.layout.run) - 1));
    const auto physical = Offsets(packet);
    const auto active = Offsets(environment & packet);
    const auto stride = std::max(result.layout.environment_size, 8U);
    result.indices.resize(8 * stride);
    for (uint32_t low = 0; low < 8; ++low) {
        for (uint32_t i = 0; i < active.size(); ++i) {
            const auto address = active[i] | (low & kept_mask);
            result.indices[low * stride + i]
                = std::distance(physical.begin(), std::lower_bound(physical.begin(), physical.end(), address))
                  * sizeof(float);
        }
    }
    return result;
}
}  // namespace mindquantum::sim::ascend
