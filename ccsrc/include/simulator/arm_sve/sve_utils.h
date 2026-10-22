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

#ifndef INCLUDE_SIMULATOR_ARM_SVE_SVE_UTILS_H_
#define INCLUDE_SIMULATOR_ARM_SVE_SVE_UTILS_H_

#include <arm_sve.h>

#include <complex>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

#include "config/config.h"
#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "simulator/state_index.h"

// Thin wrappers over the SVE ACLE intrinsics.  The single-instruction
// wrappers are MQ_ALWAYS_INLINE: they are called once per vector of
// amplitudes inside the simulator kernels and exist only to be flattened into
// those loops, so their inlining must not depend on the translation unit's
// inline budget (see MQ_ALWAYS_INLINE).  Loops, scratch management and the
// traversal drivers below are ordinary functions.
namespace mindquantum::sim::sve {
template <typename calc_type>
struct Traits;

template <>
struct Traits<float> {
    using vector_t = svfloat32_t;
    using index_vector_t = svuint32_t;
    using index_scalar_t = std::uint32_t;

    MQ_ALWAYS_INLINE static auto TruePredicate() -> svbool_t {
        return svptrue_b32();
    }
    MQ_ALWAYS_INLINE static auto ScalarLanes() -> size_t {
        return svcntw();
    }
    MQ_ALWAYS_INLINE static auto LaneIndex() -> index_vector_t {
        return svindex_u32(0, 1);
    }
    MQ_ALWAYS_INLINE static auto Duplicate(float value) -> vector_t {
        return svdup_n_f32(value);
    }
    MQ_ALWAYS_INLINE static auto DuplicateIndex(index_scalar_t value) -> index_vector_t {
        return svdup_n_u32(value);
    }
};

template <>
struct Traits<double> {
    using vector_t = svfloat64_t;
    using index_vector_t = svuint64_t;
    using index_scalar_t = std::uint64_t;

    MQ_ALWAYS_INLINE static auto TruePredicate() -> svbool_t {
        return svptrue_b64();
    }
    MQ_ALWAYS_INLINE static auto ScalarLanes() -> size_t {
        return svcntd();
    }
    MQ_ALWAYS_INLINE static auto LaneIndex() -> index_vector_t {
        return svindex_u64(0, 1);
    }
    MQ_ALWAYS_INLINE static auto Duplicate(double value) -> vector_t {
        return svdup_n_f64(value);
    }
    MQ_ALWAYS_INLINE static auto DuplicateIndex(index_scalar_t value) -> index_vector_t {
        return svdup_n_u64(value);
    }
};

template <typename calc_type>
using vector_t = typename Traits<calc_type>::vector_t;

template <typename calc_type>
using index_vector_t = typename Traits<calc_type>::index_vector_t;

template <typename calc_type>
MQ_ALWAYS_INLINE auto TruePredicate() -> svbool_t {
    return Traits<calc_type>::TruePredicate();
}

MQ_ALWAYS_INLINE auto FalsePredicate() -> svbool_t {
    return svpfalse_b();
}

template <typename calc_type>
inline constexpr auto LocalStateIndexBits() noexcept -> unsigned {
    using index_scalar_t = typename Traits<calc_type>::index_scalar_t;
    static_assert(std::is_unsigned_v<index_scalar_t>);
    return std::numeric_limits<index_scalar_t>::digits - 1;
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Zero() -> vector_t<calc_type> {
    return Traits<calc_type>::Duplicate(calc_type{0});
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto DuplicateIndex(index_t value) -> index_vector_t<calc_type> {
    using index_scalar_t = typename Traits<calc_type>::index_scalar_t;
    return Traits<calc_type>::DuplicateIndex(static_cast<index_scalar_t>(value));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto LaneIndex() -> index_vector_t<calc_type> {
    return Traits<calc_type>::LaneIndex();
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Add(svbool_t pg, vector_t<calc_type> lhs, vector_t<calc_type> rhs) -> vector_t<calc_type> {
    return svadd_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Subtract(svbool_t pg, vector_t<calc_type> lhs, vector_t<calc_type> rhs) -> vector_t<calc_type> {
    return svsub_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Negate(svbool_t pg, vector_t<calc_type> value) -> vector_t<calc_type> {
    return svneg_x(pg, value);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Multiply(svbool_t pg, vector_t<calc_type> lhs, vector_t<calc_type> rhs) -> vector_t<calc_type> {
    return svmul_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Multiply(svbool_t pg, vector_t<calc_type> value, calc_type scalar) -> vector_t<calc_type> {
    return Multiply<calc_type>(pg, value, Traits<calc_type>::Duplicate(scalar));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto Select(svbool_t pg, vector_t<calc_type> when_true, vector_t<calc_type> when_false)
    -> vector_t<calc_type> {
    return svsel(pg, when_true, when_false);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ReduceAdd(svbool_t pg, vector_t<calc_type> value) -> calc_type {
    return svaddv(pg, value);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto AddIndex(svbool_t pg, index_vector_t<calc_type> lhs, index_vector_t<calc_type> rhs)
    -> index_vector_t<calc_type> {
    return svadd_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto AddIndex(svbool_t pg, index_vector_t<calc_type> value, index_t scalar)
    -> index_vector_t<calc_type> {
    return AddIndex<calc_type>(pg, value, DuplicateIndex<calc_type>(scalar));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ShiftLeftIndex(svbool_t pg, index_vector_t<calc_type> value, unsigned amount)
    -> index_vector_t<calc_type> {
    return svlsl_x(pg, value, DuplicateIndex<calc_type>(amount));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ShiftRightIndex(svbool_t pg, index_vector_t<calc_type> value, unsigned amount)
    -> index_vector_t<calc_type> {
    return svlsr_x(pg, value, DuplicateIndex<calc_type>(amount));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto AndIndex(svbool_t pg, index_vector_t<calc_type> lhs, index_vector_t<calc_type> rhs)
    -> index_vector_t<calc_type> {
    return svand_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto AndIndex(svbool_t pg, index_vector_t<calc_type> value, index_t mask)
    -> index_vector_t<calc_type> {
    return AndIndex<calc_type>(pg, value, DuplicateIndex<calc_type>(mask));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto XorIndex(svbool_t pg, index_vector_t<calc_type> lhs, index_vector_t<calc_type> rhs)
    -> index_vector_t<calc_type> {
    return sveor_x(pg, lhs, rhs);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto SelectIndex(svbool_t pg, index_vector_t<calc_type> when_true,
                                  index_vector_t<calc_type> when_false) -> index_vector_t<calc_type> {
    return svsel(pg, when_true, when_false);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto IndexLessThan(index_vector_t<calc_type> value, index_t limit) -> svbool_t {
    return svcmplt(TruePredicate<calc_type>(), value, DuplicateIndex<calc_type>(limit));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto IndexEquals(svbool_t pg, index_vector_t<calc_type> value, index_t expected) -> svbool_t {
    return svcmpeq(pg, value, DuplicateIndex<calc_type>(expected));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto IndexNotEquals(svbool_t pg, index_vector_t<calc_type> value, index_t expected) -> svbool_t {
    return svcmpne(pg, value, DuplicateIndex<calc_type>(expected));
}

template <typename calc_type>
inline auto ComplexStep() -> omp::idx_t {
    return static_cast<omp::idx_t>(Traits<calc_type>::ScalarLanes() / 2);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto LoopIndex(index_t block_start) -> index_vector_t<calc_type> {
    const auto pg = TruePredicate<calc_type>();
    return AddIndex<calc_type>(pg, ShiftRightIndex<calc_type>(pg, LaneIndex<calc_type>(), 1), block_start);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ActiveComplexPredicate(index_vector_t<calc_type> loop_index, index_t block_count) -> svbool_t {
    return IndexLessThan<calc_type>(loop_index, block_count);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ComplexScalarOffset(svbool_t pg, index_vector_t<calc_type> complex_index)
    -> index_vector_t<calc_type> {
    return AddIndex<calc_type>(pg, ShiftLeftIndex<calc_type>(pg, complex_index, 1),
                               AndIndex<calc_type>(pg, LaneIndex<calc_type>(), 1));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto GatherComplex(const calc_type* base, svbool_t pg, index_vector_t<calc_type> complex_index)
    -> vector_t<calc_type> {
    return svld1_gather_index(pg, base, ComplexScalarOffset<calc_type>(pg, complex_index));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto GatherComplexAt(const std::complex<calc_type>* state, index_t fixed_offset, svbool_t pg,
                            index_vector_t<calc_type> local_index) -> vector_t<calc_type> {
    return GatherComplex<calc_type>(reinterpret_cast<const calc_type*>(state + fixed_offset), pg, local_index);
}

template <typename calc_type>
MQ_ALWAYS_INLINE void ScatterComplex(calc_type* base, svbool_t pg, index_vector_t<calc_type> complex_index,
                           vector_t<calc_type> value) {
    svst1_scatter_index(pg, base, ComplexScalarOffset<calc_type>(pg, complex_index), value);
}

template <typename calc_type>
MQ_ALWAYS_INLINE void ScatterComplexAt(std::complex<calc_type>* state, index_t fixed_offset, svbool_t pg,
                             index_vector_t<calc_type> local_index, vector_t<calc_type> value) {
    ScatterComplex<calc_type>(reinterpret_cast<calc_type*>(state + fixed_offset), pg, local_index, value);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto BroadcastComplex(std::complex<calc_type> value) -> vector_t<calc_type> {
    const auto pg = TruePredicate<calc_type>();
    const auto real_lane = IndexEquals<calc_type>(pg, AndIndex<calc_type>(pg, LaneIndex<calc_type>(), 1), 0);
    return Select<calc_type>(real_lane, Traits<calc_type>::Duplicate(value.real()),
                             Traits<calc_type>::Duplicate(value.imag()));
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ComplexMulAdd(svbool_t pg, vector_t<calc_type> accumulator, vector_t<calc_type> value,
                          vector_t<calc_type> coefficient) -> vector_t<calc_type> {
    accumulator = svcmla_x(pg, accumulator, value, coefficient, 0);
    return svcmla_x(pg, accumulator, value, coefficient, 90);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto CoefficientMulAdd(svbool_t pg, vector_t<calc_type> accumulator, vector_t<calc_type> value,
                              std::complex<calc_type> coefficient) -> vector_t<calc_type> {
    if (coefficient == std::complex<calc_type>{0, 0}) {
        return accumulator;
    }
    const auto broadcast = BroadcastComplex<calc_type>(coefficient);
    if (coefficient.imag() == calc_type{0}) {
        return svcmla_x(pg, accumulator, broadcast, value, 0);
    }
    if (coefficient.real() == calc_type{0}) {
        return svcmla_x(pg, accumulator, broadcast, value, 90);
    }
    return ComplexMulAdd<calc_type>(pg, accumulator, value, broadcast);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ComplexMultiply(svbool_t pg, vector_t<calc_type> value, std::complex<calc_type> coefficient)
    -> vector_t<calc_type> {
    return CoefficientMulAdd<calc_type>(pg, Zero<calc_type>(), value, coefficient);
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto ComplexMultiply(svbool_t pg, vector_t<calc_type> value, vector_t<calc_type> coefficient)
    -> vector_t<calc_type> {
    return ComplexMulAdd<calc_type>(pg, Zero<calc_type>(), value, coefficient);
}

template <typename calc_type>
inline auto ExpandBaseIndex(svbool_t pg, index_vector_t<calc_type> compressed, const qbits_t& qubits)
    -> index_vector_t<calc_type> {
    auto base = compressed;
    for (auto qubit : qubits) {
        const auto low_mask = (static_cast<index_t>(1) << qubit) - 1;
        base = AddIndex<calc_type>(pg, ShiftLeftIndex<calc_type>(pg, AndIndex<calc_type>(pg, base, ~low_mask), 1),
                                   AndIndex<calc_type>(pg, base, low_mask));
    }
    return base;
}

template <typename calc_type>
MQ_ALWAYS_INLINE auto LoadCachedComplex(const calc_type* cache, size_t basis) -> vector_t<calc_type> {
    return svld1(TruePredicate<calc_type>(), cache + basis * Traits<calc_type>::ScalarLanes());
}

template <typename calc_type>
MQ_ALWAYS_INLINE void StoreCachedComplex(calc_type* cache, size_t basis, svbool_t pg, vector_t<calc_type> value) {
    svst1(pg, cache + basis * Traits<calc_type>::ScalarLanes(), value);
}

template <typename calc_type>
inline auto AlignScratch(std::vector<calc_type>* storage, size_t scratch_size) -> calc_type* {
    const auto alignment = static_cast<uintptr_t>(svcntb());
    const auto address = reinterpret_cast<uintptr_t>(storage->data());
    const auto remainder = address % alignment;
    const auto aligned_address = remainder == 0 ? address : address + alignment - remainder;
    auto* scratch = reinterpret_cast<calc_type*>(aligned_address);
    if (static_cast<size_t>(scratch - storage->data()) + scratch_size > storage->size()) {
        throw std::bad_alloc();
    }
    return scratch;
}

template <typename calc_type>
inline auto ThreadScratch(size_t scratch_size) -> calc_type* {
    thread_local std::vector<calc_type> storage;
    const auto required_size = scratch_size + Traits<calc_type>::ScalarLanes();
    if (storage.size() < required_size) {
        storage.resize(required_size);
    }
    return AlignScratch(&storage, scratch_size);
}

template <typename calc_type, typename kernel_t>
void ForEachStateIndexBatchInParallel(const state_index::StateIndexWindow& index_window, index_t compressed_count,
                                      const kernel_t& kernel) {
    const auto step = ComplexStep<calc_type>();
    if (compressed_count <= index_window.WindowCapacity()) {
        MQ_DO_PRAGMA(omp for schedule(static))
        for (omp::idx_t compressed_start = 0; compressed_start < static_cast<omp::idx_t>(compressed_count);
             compressed_start += step) {
            kernel(state_index::StateIndexBatch{0, static_cast<index_t>(compressed_start), compressed_count}, 0);
        }
        return;
    }

    const state_index::StateIndexBatchTraversal traversal(index_window, compressed_count, static_cast<index_t>(step));
    state_index::StateIndexWindowCursor cursor(index_window);
    MQ_DO_PRAGMA(omp for schedule(static))
    for (omp::idx_t ordinal = 0; ordinal < static_cast<omp::idx_t>(traversal.BatchCount()); ++ordinal) {
        const auto batch = traversal.Batch(static_cast<index_t>(ordinal));
        kernel(batch, cursor.StateOrigin(batch.compressed_window_origin));
    }
}

template <typename calc_type, typename kernel_t>
void ForEachStateIndexBatchSerial(const state_index::StateIndexWindow& index_window, index_t compressed_count,
                                  const kernel_t& kernel) {
    const auto step = ComplexStep<calc_type>();
    if (compressed_count <= index_window.WindowCapacity()) {
        for (omp::idx_t compressed_start = 0; compressed_start < static_cast<omp::idx_t>(compressed_count);
             compressed_start += step) {
            kernel(state_index::StateIndexBatch{0, static_cast<index_t>(compressed_start), compressed_count}, 0);
        }
        return;
    }

    const state_index::StateIndexBatchTraversal traversal(index_window, compressed_count, static_cast<index_t>(step));
    state_index::StateIndexWindowCursor cursor(index_window);
    for (index_t ordinal = 0; ordinal < traversal.BatchCount(); ++ordinal) {
        const auto batch = traversal.Batch(ordinal);
        kernel(batch, cursor.StateOrigin(batch.compressed_window_origin));
    }
}

template <typename calc_type, typename kernel_t>
void ParallelForEachStateIndexBatchWithScratch(int threads, const state_index::StateIndexWindow& index_window,
                                               index_t compressed_count, size_t scratch_vectors,
                                               const kernel_t& kernel) {
    const auto scratch_size = scratch_vectors * Traits<calc_type>::ScalarLanes();
    MQ_DO_PRAGMA(omp parallel num_threads(threads)) {
        auto* scratch = ThreadScratch<calc_type>(scratch_size);
        ForEachStateIndexBatchInParallel<calc_type>(
            index_window, compressed_count, [&](const state_index::StateIndexBatch& batch, index_t state_origin) {
                kernel(batch, state_origin, scratch);
            });
    }
}

template <typename calc_type, typename kernel_t>
void ForEachStateIndexBatchWithScratch(index_t dim, const state_index::StateIndexWindow& index_window,
                                       index_t compressed_count, size_t scratch_vectors, const kernel_t& kernel) {
    if (const int threads = omp::TeamSize(dim); threads != 1) {
        ParallelForEachStateIndexBatchWithScratch<calc_type>(threads, index_window, compressed_count, scratch_vectors,
                                                             kernel);
        return;
    }

    const auto scratch_size = scratch_vectors * Traits<calc_type>::ScalarLanes();
    auto* scratch = ThreadScratch<calc_type>(scratch_size);
    ForEachStateIndexBatchSerial<calc_type>(
        index_window, compressed_count,
        [&](const state_index::StateIndexBatch& batch, index_t state_origin) { kernel(batch, state_origin, scratch); });
}
}  // namespace mindquantum::sim::sve

#endif  // INCLUDE_SIMULATOR_ARM_SVE_SVE_UTILS_H_
