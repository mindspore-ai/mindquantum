/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_BASIS_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_BASIS_H

#include <cmath>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/utils.h"

namespace mindquantum::sim::chem::detail {

// Nested namespace for CI-basis data structures
namespace ci_basis {

class Combinatorics {
 public:
    //! Get the number of combinations C(n, k)
    static uint64_t get(int n, int k) {
        std::call_once(init_flag_, &Combinatorics::initialize);
        if (k < 0 || n < 0 || k > n || n >= max_n_) {
            return 0;
        }
        return table_[n][k];
    }

 private:
    static void initialize() {
        table_.assign(max_n_ + 1, std::vector<uint64_t>(max_n_ + 1, 0));
        for (int i = 0; i <= max_n_; ++i) {
            table_[i][0] = 1;
            for (int j = 1; j <= i; ++j) {
                table_[i][j] = table_[i - 1][j - 1] + table_[i - 1][j];
            }
        }
    }
    static constexpr int max_n_ = 64;
    static inline std::vector<std::vector<uint64_t>> table_;
    static inline std::once_flag init_flag_;
};

//! Convert a lexicographical index into a bitmask for n_electrons in n_qubits.
inline uint64_t unrank_lexicographical(size_t idx, int n_qubits, int n_electrons) {
    uint64_t mask = 0;
    int rem = n_electrons;
    for (int i = n_qubits - 1; i >= 0 && rem > 0; --i) {
        size_t c = Combinatorics::get(i, rem);
        if (idx >= c) {
            idx -= c;
            mask |= (1ULL << i);
            --rem;
        }
    }
    return mask;
}

class Indexing {
 public:
    Indexing(int n_qubits, int n_electrons) : n_qubits_(n_qubits), n_electrons_(n_electrons) {
        initialize();
    }

    uint64_t unrank(size_t index) const {
        return index_to_mask_[index];
    }

    size_t rank(uint64_t mask) const {
        return mask_to_index_[mask];
    }

 private:
    void initialize() {
        size_t dim = Combinatorics::get(n_qubits_, n_electrons_);
        index_to_mask_.resize(dim);
        mask_to_index_.resize(1ULL << n_qubits_);
        THRESHOLD_OMP_FOR(
            dim, 1UL << 13, for (omp::idx_t i = 0; i < dim; ++i) {
                uint64_t mask = unrank_lexicographical(i, n_qubits_, n_electrons_);
                index_to_mask_[i] = mask;
                mask_to_index_[mask] = i;
            });
    }

    int n_qubits_;
    int n_electrons_;
    std::vector<uint64_t> index_to_mask_;
    std::vector<size_t> mask_to_index_;
};

class IndexingManager {
 public:
    static std::shared_ptr<Indexing> GetIndexer(int n_qubits, int n_electrons) {
        static IndexingManager instance;
        std::lock_guard<std::mutex> lock(instance.mutex_);
        auto key = std::make_pair(n_qubits, n_electrons);
        auto it = instance.indexers_.find(key);
        if (it == instance.indexers_.end()) {
            auto indexer = std::make_shared<Indexing>(n_qubits, n_electrons);
            instance.indexers_[key] = indexer;
            return indexer;
        }
        return it->second;
    }

 private:
    IndexingManager() = default;
    std::map<std::pair<int, int>, std::shared_ptr<Indexing>> indexers_;
    std::mutex mutex_;
};

// Representation of a Slater determinant via bitmask of occupied spin-orbitals
struct SlaterDeterminant {
    uint64_t occupied_orbitals;
    int n_electrons;
    int n_qubits;

    SlaterDeterminant(uint64_t mask, int n_elec, int n_qbt)
        : occupied_orbitals(mask), n_electrons(n_elec), n_qubits(n_qbt) {
    }

    // Hartree-Fock reference state: lowest n_electrons orbitals occupied
    static SlaterDeterminant HartreeFock(int n_qubits, int n_electrons) {
        uint64_t mask = (n_electrons >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n_electrons) - 1);
        return SlaterDeterminant(mask, n_electrons, n_qubits);
    }

    bool is_occupied(int orbital_idx) const {
        return ((occupied_orbitals >> orbital_idx) & 1ULL) != 0ULL;
    }

    // Count set bits using a compiler intrinsic for performance.
    // For C++20, this could be replaced with std::popcount.
    static int count_set_bits(uint64_t n) {
        return static_cast<int>(__builtin_popcountll(n));
    }

    bool operator<(const SlaterDeterminant& other) const {
        if (occupied_orbitals != other.occupied_orbitals) {
            return occupied_orbitals < other.occupied_orbitals;
        }
        if (n_electrons != other.n_electrons) {
            return n_electrons < other.n_electrons;
        }
        return n_qubits < other.n_qubits;
    }

    bool operator==(const SlaterDeterminant& other) const {
        return occupied_orbitals == other.occupied_orbitals && n_electrons == other.n_electrons
               && n_qubits == other.n_qubits;
    }

    // Binary string representation, e.g., |1100>
    std::string to_string() const {
        std::ostringstream oss;
        oss << "|";
        for (int i = n_qubits - 1; i >= 0; --i) {
            oss << (((occupied_orbitals >> i) & 1ULL) ? '1' : '0');
        }
        oss << ">";
        return oss.str();
    }
};

template <typename calc_t>
class CIVector {
 public:
    using basis_t = SlaterDeterminant;
    using calc_type = calc_t;

    /// Construct a zero vector for given qubit and electron counts
    CIVector(int n_qubits, int n_electrons)
        : n_qubits_(n_qubits), n_electrons_(n_electrons), data_(Combinatorics::get(n_qubits, n_electrons), calc_t(0)) {
    }

    /// Number of qubits (spin-orbitals)
    int n_qubits() const noexcept {
        return n_qubits_;
    }
    /// Number of electrons (occupied orbitals per determinant)
    int n_electrons() const noexcept {
        return n_electrons_;
    }
    /// Total basis dimension = C(n_qubits, n_electrons)
    size_t dimension() const noexcept {
        return data_.size();
    }

    /// Raw contiguous amplitude storage
    const std::vector<calc_t>& data() const noexcept {
        return data_;
    }
    std::vector<calc_t>& data() noexcept {
        return data_;
    }

    /// Indexing into the dense amplitude array
    calc_t operator[](size_t idx) const noexcept {
        return data_[idx];
    }
    calc_t& operator[](size_t idx) noexcept {
        return data_[idx];
    }

    /// Reset all amplitudes to zero
    void clear() noexcept {
        std::fill(data_.begin(), data_.end(), calc_t(0));
    }

    /// Normalize the vector (L2 norm)
    void normalize() {
        double norm2 = 0;
        for (auto& v : data_) {
            norm2 += std::norm(v);
        }
        norm2 = std::sqrt(norm2);
        if (norm2 <= 0) {
            return;
        }
        for (auto& v : data_) {
            v /= static_cast<calc_t>(norm2);
        }
    }

    /// Add amplitude to a given index, pruning near-zero results
    void add_amplitude_by_index(size_t index, calc_t amplitude, double tolerance = 1e-9) {
        data_[index] += amplitude;
        if (std::abs(data_[index]) < tolerance) {
            data_[index] = calc_t(0);
        }
    }

    /// Retrieve amplitude for a given index
    calc_t get_amplitude_by_index(size_t index) const {
        return data_[index];
    }

    /// Count non-zero amplitudes in the vector
    size_t non_zero_count() const noexcept {
        size_t cnt = 0;
        for (auto const& v : data_) {
            if (v != calc_t(0)) {
                ++cnt;
            }
        }
        return cnt;
    }

 private:
    int n_qubits_;
    int n_electrons_;
    std::vector<calc_t> data_;
};

}  // namespace ci_basis

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CI_BASIS_H
