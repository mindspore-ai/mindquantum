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

#ifndef INCLUDE_SIMULATOR_VECTOR_DETAIL_STATE_INDEX_H_
#define INCLUDE_SIMULATOR_VECTOR_DETAIL_STATE_INDEX_H_

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/mq_base_types.h"

namespace mindquantum::sim::vector::detail::state_index {
class BitExpander {
 public:
    BitExpander() = default;

    explicit BitExpander(const qbits_t& excluded_qubits) {
        low_masks_.reserve(excluded_qubits.size());
        qbit_t previous = -1;
        for (auto qubit : excluded_qubits) {
            if (qubit < 0 || qubit >= static_cast<qbit_t>(std::numeric_limits<index_t>::digits) || qubit <= previous) {
                throw std::invalid_argument("Excluded qubits must be strictly increasing valid index bits.");
            }
            low_masks_.push_back((index_t{1} << static_cast<unsigned>(qubit)) - 1);
            previous = qubit;
        }
    }

    [[nodiscard]] auto Expand(index_t compressed) const noexcept -> index_t {
        for (auto low_mask : low_masks_) {
            compressed = ((compressed & ~low_mask) << 1) | (compressed & low_mask);
        }
        return compressed;
    }

 private:
    std::vector<index_t> low_masks_;
};

// A vector gather index is deliberately local to one state window.  The
// scalar window origin retains every higher state-index bit, so an instruction
// encoding limit never becomes a simulator-dimension limit.
class StateIndexWindow {
 public:
    StateIndexWindow(unsigned local_state_bits, const qbits_t& excluded_qubits)
        : local_state_bits_(ValidateLocalStateBits(local_state_bits)), global_expander_(excluded_qubits) {
        local_excluded_qubits_.reserve(excluded_qubits.size());
        std::copy_if(excluded_qubits.begin(), excluded_qubits.end(), std::back_inserter(local_excluded_qubits_),
                     [local_state_bits](qbit_t qubit) { return qubit < static_cast<qbit_t>(local_state_bits); });
        local_expander_ = BitExpander(local_excluded_qubits_);

        const auto local_compressed_bits = local_state_bits_ - local_excluded_qubits_.size();
        window_capacity_ = index_t{1} << local_compressed_bits;
        window_mask_ = window_capacity_ - 1;
        local_state_mask_ = (index_t{1} << local_state_bits_) - 1;
    }

    [[nodiscard]] auto WindowCapacity() const noexcept -> index_t {
        return window_capacity_;
    }

    [[nodiscard]] auto LocalStateMask() const noexcept -> index_t {
        return local_state_mask_;
    }

    [[nodiscard]] auto LocalExcludedQubits() const noexcept -> const qbits_t& {
        return local_excluded_qubits_;
    }

    [[nodiscard]] auto CompressedWindowOrigin(index_t compressed) const noexcept -> index_t {
        return compressed & ~window_mask_;
    }

    [[nodiscard]] auto LocalCompressedIndex(index_t compressed) const noexcept -> index_t {
        return compressed & window_mask_;
    }

    [[nodiscard]] auto StateWindowOrigin(index_t compressed_window_origin) const noexcept -> index_t {
        assert(LocalCompressedIndex(compressed_window_origin) == 0);
        return global_expander_.Expand(compressed_window_origin);
    }

    [[nodiscard]] auto Expand(index_t compressed) const noexcept -> index_t {
        return global_expander_.Expand(compressed);
    }

    [[nodiscard]] auto ExpandLocal(index_t local_compressed) const noexcept -> index_t {
        return local_expander_.Expand(local_compressed);
    }

 private:
    static auto ValidateLocalStateBits(unsigned local_state_bits) -> unsigned {
        if (local_state_bits == 0 || local_state_bits >= std::numeric_limits<index_t>::digits) {
            throw std::invalid_argument("Local state-index width must be smaller than the global index width.");
        }
        return local_state_bits;
    }

    unsigned local_state_bits_;
    BitExpander global_expander_;
    BitExpander local_expander_;
    qbits_t local_excluded_qubits_;
    index_t window_capacity_;
    index_t window_mask_;
    index_t local_state_mask_;
};

struct StateIndexBatch {
    index_t compressed_window_origin;
    index_t local_compressed_start;
    index_t local_compressed_count;
};

// Flattens all vector batches while guaranteeing that no batch crosses a
// window.  When the vector width divides the window, ordinal-to-index mapping
// is multiplication plus bit masking only.  The general mapping is required
// for legal SVE vector lengths such as 384 bits, whose complex-lane count need
// not divide a power-of-two window.
class StateIndexBatchTraversal {
 public:
    StateIndexBatchTraversal(const StateIndexWindow& window, index_t compressed_count, index_t batch_width)
        : window_(window), compressed_count_(compressed_count), batch_width_(batch_width) {
        if (compressed_count_ == 0) {
            throw std::invalid_argument("State-index traversal cannot be empty.");
        }
        if (batch_width_ == 0) {
            throw std::invalid_argument("State-index traversal batch width cannot be zero.");
        }

        const auto capacity = window_.WindowCapacity();
        batch_width_divides_window_ = capacity % batch_width_ == 0;
        batches_per_window_ = CeilDiv(capacity, batch_width_);
        full_window_count_ = compressed_count_ / capacity;
        const auto tail_count = compressed_count_ % capacity;
        batch_count_ = batch_width_divides_window_
                           ? CeilDiv(compressed_count_, batch_width_)
                           : full_window_count_ * batches_per_window_ + CeilDiv(tail_count, batch_width_);
    }

    [[nodiscard]] auto BatchCount() const noexcept -> index_t {
        return batch_count_;
    }

    [[nodiscard]] auto Batch(index_t ordinal) const noexcept -> StateIndexBatch {
        assert(ordinal < batch_count_);
        if (batch_width_divides_window_) {
            const auto compressed_start = ordinal * batch_width_;
            const auto window_origin = window_.CompressedWindowOrigin(compressed_start);
            return MakeBatch(window_origin, window_.LocalCompressedIndex(compressed_start));
        }

        const auto full_window_batch_count = full_window_count_ * batches_per_window_;
        const auto window_ordinal = ordinal < full_window_batch_count ? ordinal / batches_per_window_
                                                                      : full_window_count_;
        const auto local_batch_ordinal = ordinal < full_window_batch_count ? ordinal % batches_per_window_
                                                                           : ordinal - full_window_batch_count;
        return MakeBatch(window_ordinal * window_.WindowCapacity(), local_batch_ordinal * batch_width_);
    }

 private:
    static constexpr auto CeilDiv(index_t numerator, index_t denominator) noexcept -> index_t {
        return numerator / denominator + static_cast<index_t>(numerator % denominator != 0);
    }

    [[nodiscard]] auto MakeBatch(index_t window_origin, index_t local_start) const noexcept -> StateIndexBatch {
        const auto remaining = compressed_count_ - window_origin;
        return {window_origin, local_start, std::min(window_.WindowCapacity(), remaining)};
    }

    const StateIndexWindow& window_;
    index_t compressed_count_;
    index_t batch_width_;
    bool batch_width_divides_window_;
    index_t batches_per_window_;
    index_t full_window_count_;
    index_t batch_count_;
};

class StateIndexWindowCursor {
 public:
    explicit StateIndexWindowCursor(const StateIndexWindow& window) : window_(window) {
    }

    [[nodiscard]] auto StateOrigin(index_t compressed_window_origin) noexcept -> index_t {
        if (!initialized_ || compressed_window_origin != compressed_window_origin_) {
            compressed_window_origin_ = compressed_window_origin;
            state_origin_ = window_.StateWindowOrigin(compressed_window_origin);
            initialized_ = true;
        }
        return state_origin_;
    }

 private:
    const StateIndexWindow& window_;
    bool initialized_ = false;
    index_t compressed_window_origin_ = 0;
    index_t state_origin_ = 0;
};
}  // namespace mindquantum::sim::vector::detail::state_index

#endif  // INCLUDE_SIMULATOR_VECTOR_DETAIL_STATE_INDEX_H_
