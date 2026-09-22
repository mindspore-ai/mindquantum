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
#ifndef INCLUDE_SIMULATOR_VECTOR_ASCEND_OBSERVABLE_H_
#define INCLUDE_SIMULATOR_VECTOR_ASCEND_OBSERVABLE_H_

#include <stdint.h>

#include <acl/acl.h>

// The .asc translation unit needs only the plain host/device ABI below.
#ifndef MQ_ASCEND_OBSERVABLE_DEVICE_ONLY
#    include <vector>

namespace mindquantum {
template <typename T>
struct Hamiltonian;
}
#endif

namespace mindquantum::sim::ascend {
constexpr uint32_t kObservableTileSize = 1024;
constexpr uint32_t kObservableMaxBlocks = 48;
constexpr uint32_t kObservablePauli = 0;
constexpr uint32_t kObservableCsr = 1;
constexpr uint64_t kInvalidSampleIndex = UINT64_MAX;

struct ObservableValue {
    float real;
    float imag;
};

struct alignas(32) ObservableLayout {
    uint64_t dimension;  // 2^n, n in [0, 32]
    uint64_t plane;      // max(8, dimension), in float elements
    uint64_t group_count;
    uint32_t kind;
    uint32_t tile_size;  // min(plane, kObservableTileSize)
};

struct alignas(32) ObservableGroup {
    uint64_t flip_mask;
    uint64_t term_begin;
    uint64_t term_count;
    uint32_t gather_offset;  // uint32 elements in gather_indices(), not bytes
    uint32_t reserved;
};

struct alignas(32) ObservableTerm {
    uint64_t parity_mask;  // Y | Z, in OUTPUT basis; high bits handled on device
    uint32_t sign_offset;  // float elements in signs()
    uint32_t reserved;
    ObservableValue coefficient;  // c * (-i)^number_of_Y
    uint64_t reserved2;
};

// One independently written, DMA-aligned record per launched AIV, even when
// that AIV has no work. Combine in HOST DOUBLE as sum(real + real_correction)
// and sum(imag + imag_correction), converting each operand before addition.
// Kahan accumulation combines balanced per-tile ReduceSum results on device.
struct alignas(32) ReductionPartial {
    float real;
    float imag;
    float real_correction;
    float imag_correction;
    float reserved[4];
};

struct alignas(32) SamplingLayout {
    uint64_t dimension;
    uint64_t plane;
    uint32_t bins;           // 1..48, also the number of AIVs building the tree
    uint32_t tiles_per_bin;  // power of two, >=8; returned by MakeSamplingLayout
    uint64_t reserved;
};

struct alignas(32) SamplingRequest {
    uint32_t bin;
    float fraction;  // conditional uniform variate in [0,1), NOT an absolute probability
    uint64_t reserved[3];
};

struct alignas(32) SamplingResult {
    uint64_t index;  // full computational-basis index; filter measured bits on host
    uint64_t reserved[3];
};

struct alignas(32) DensityLayout {
    uint64_t dimension;
    uint64_t kept_mask;
    uint64_t external_mask;
    uint64_t tile_count;
    uint64_t output_dimension;
    uint32_t packet_size;
    uint32_t run;
    uint32_t environment_size;
    uint32_t shards;
};

static_assert(sizeof(ObservableValue) == 8, "Ascend observable value ABI mismatch");
static_assert(sizeof(ObservableLayout) == 32, "Ascend observable layout ABI mismatch");
static_assert(sizeof(ObservableGroup) == 32, "Ascend observable group ABI mismatch");
static_assert(sizeof(ObservableTerm) == 32, "Ascend observable term ABI mismatch");
static_assert(sizeof(ReductionPartial) == 32, "Ascend reduction ABI mismatch");
static_assert(sizeof(SamplingLayout) == 32, "Ascend sampling layout ABI mismatch");
static_assert(sizeof(SamplingRequest) == 32, "Ascend sampling request ABI mismatch");
static_assert(sizeof(SamplingResult) == 32, "Ascend sampling result ABI mismatch");
static_assert(sizeof(DensityLayout) == 64, "Ascend reduced-density layout ABI mismatch");

#ifndef MQ_ASCEND_OBSERVABLE_DEVICE_ONLY
// Pure, validated, immutable snapshot. ORIGIN/BACKEND compile ham_, never the
// BACKEND half-diagonal/upper CSR. FRONTEND is a full, possibly non-Hermitian
// complex CSR. Duplicate CSR columns are legal and are added, not overwritten.
// Pauli words must have distinct in-range qubits and axes I/X/Y/Z. ORIGIN's
// n_qubits_ is not authoritative; other modes must match the supplied n_qubits.
// CSR raw pointers must own the lengths declared by dim_/nnz_; C++ cannot
// discover their allocation lengths. All structural contents are validated.
class ObservableProgram {
 public:
    static ObservableProgram Compile(const Hamiltonian<float>& hamiltonian, int64_t n_qubits);

    const ObservableLayout& layout() const {
        return layout_;
    }
    const std::vector<ObservableGroup>& groups() const {
        return groups_;
    }
    const std::vector<ObservableTerm>& terms() const {
        return terms_;
    }
    const std::vector<uint32_t>& gather_indices() const {
        return gather_indices_;
    }
    const std::vector<float>& signs() const {
        return signs_;
    }
    const std::vector<uint64_t>& row_offsets() const {
        return row_offsets_;
    }
    const std::vector<uint64_t>& columns() const {
        return columns_;
    }
    const std::vector<ObservableValue>& values() const {
        return values_;
    }

 private:
    ObservableProgram(ObservableLayout layout, std::vector<ObservableGroup> groups, std::vector<ObservableTerm> terms,
                      std::vector<uint32_t> gather_indices, std::vector<float> signs, std::vector<uint64_t> row_offsets,
                      std::vector<uint64_t> columns, std::vector<ObservableValue> values);
    const ObservableLayout layout_;
    const std::vector<ObservableGroup> groups_;
    const std::vector<ObservableTerm> terms_;
    const std::vector<uint32_t> gather_indices_;
    const std::vector<float> signs_;
    const std::vector<uint64_t> row_offsets_;
    const std::vector<uint64_t> columns_;
    const std::vector<ObservableValue> values_;
};

SamplingLayout MakeSamplingLayout(int64_t n_qubits, uint32_t bins = kObservableMaxBlocks);

struct DensityProgram {
    DensityLayout layout;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> indices;
};
DensityProgram CompileDensity(int64_t n_qubits, uint64_t kept_mask, uint32_t cores);
#endif

// All pointers below are DEVICE pointers, base-aligned to 32 bytes. States
// contain 2*max(8,dimension) floats, real plane followed by imaginary plane.
// Input padding is ignored; state-writing kernels always zero padding. No
// implicit |0> null state, allocations, synchronizations, copies to host, or RNG
// are hidden in these launchers. blocks is in [1,48]. Callers must keep inputs
// alive and synchronize/check stream errors before consuming results.
// Invalid scalar arguments return ACL_ERROR_INVALID_PARAM without a launch;
// successful launches return aclrtGetLastError(ACL_RT_THREAD_LEVEL). Device
// metadata must come from the validated compiler; allocation lengths/partial
// overlaps cannot be checked by launchers. Compile observable.asc with
// -fno-fast-math -ffp-contract=off (fast-math is rejected at compile time).
//
// Metadata uploads use exactly vector.size()*sizeof(value_type) bytes (the
// allocator may round capacity up to 32 bytes); pools are DMA-aligned internally.
// Unused/empty metadata pointers may be null. layout is passed BY VALUE, not
// uploaded. In Pauli mode groups/terms/gather_indices/signs are used; in CSR
// mode row_offsets[dimension+1], columns[nnz], values[nnz] are used.
// output, if nonnull, is a SEPARATE state, disjoint from ket, bra and metadata.
// partials, if nonnull, requires bra and has blocks*sizeof(ReductionPartial)
// bytes. Either/both output and partials may be requested. Expectation-only
// needs no device H|ket> state; H|ket> exists only one tile at a time in UB.
aclError LaunchObservable(uint32_t blocks, aclrtStream stream, void* ket, void* bra, void* output, void* groups,
                          void* terms, void* gather_indices, void* signs, void* row_offsets, void* columns,
                          void* values, void* partials, ObservableLayout layout);

// sum_{(i & mask)==value} conj(bra[i])*ket[i]. A null bra means norm/probability
// sum |ket[i]|^2 (imaginary partials are exactly zero). mask=value=0 is the full
// inner product/norm. Probabilities are RAW MASSES on an unnormalized state:
// sample p1/norm; normalize a selected branch by 1/sqrt(raw_branch_mass).
// partials: blocks*sizeof(ReductionPartial). mask must be inside dimension-1,
// value must be a subset of mask. Dimension must be a power of two <=2^32.
aclError LaunchInnerProduct(uint32_t blocks, aclrtStream stream, void* bra, void* ket, void* partials,
                            uint64_t dimension, uint64_t mask = 0, uint64_t value = 0);

// outcome is the ORIGINAL measured bit, not the reset target. reset_to=-1
// keeps that outcome; 0/1 moves it to the requested target, using disjoint
// pair ownership so in-place high-bit reset has no inter-core read/write race.
// qubit < log2(dimension); inverse_norm must be finite and positive. The host
// rejects a zero/nonfinite branch mass before computing this normalization.
aclError LaunchCollapse(uint32_t blocks, aclrtStream stream, void* state, uint64_t dimension, uint32_t qubit,
                        uint32_t outcome, int32_t reset_to, float inverse_norm);
aclError LaunchScale(uint32_t blocks, aclrtStream stream, void* state, uint64_t dimension, float factor);

// Contract traced indices directly, never construct the full density matrix.
// Each matrix entry has layout.shards independent ReductionPartial records,
// in row-major order; combine them in host double. All state transfers use
// aligned packet DMA and Gather, including retained low-order bits.
aclError LaunchDensity(uint32_t blocks, aclrtStream stream, void* state, void* offsets, void* indices, void* partials,
                       DensityLayout layout);

// Terminal sampling, stage 1: balanced device probability tree; no amplitude
// or probability array is returned to the host. MakeSamplingLayout determines
// equal contiguous bins of tiles; trailing bins/leaves may have zero mass.
// tree: 2*bins*tiles_per_bin*sizeof(float) bytes, device-only scratch. Each bin
// has a binary heap at base=2*bin*tiles_per_bin: root [base+1], leaf t at
// [base+tiles_per_bin+t], unused [base+0]. A leaf reduces <=1024 probabilities.
// totals: bins*sizeof(ReductionPartial), real=root, all other fields zero.
// blocks MUST equal layout.bins. Rebuild after ANY state change.
aclError LaunchSamplingTree(uint32_t blocks, aclrtStream stream, void* state, void* tree, void* totals,
                            SamplingLayout layout);

// Stage 2: host double-sums the <=48 totals, rejects zero/nonfinite norm,
// selects bins proportionally with seeded RNG, and uploads conditional uniform
// fractions (clamp a float conversion of a draw below 1, never to 1).
// requests: shots*sizeof(SamplingRequest); results: shots*sizeof(SamplingResult).
// No shot sorting required. Search takes log2(tiles_per_bin) GM tree steps,
// one state tile, balanced UB subreductions, then at most eight scalar lanes.
// Empty bins/invalid request fields produce kInvalidSampleIndex. Boundary
// roundoff selects the last positive mass, never a zero-probability amplitude.
// State must be the SAME unchanged state used to build tree. Results preserve
// request order. shots=0 is a no-op kernel and permits null requests/results.
aclError LaunchSample(uint32_t blocks, aclrtStream stream, void* state, void* tree, void* requests, void* results,
                      uint64_t shots, SamplingLayout layout);
}  // namespace mindquantum::sim::ascend
#endif  // INCLUDE_SIMULATOR_VECTOR_ASCEND_OBSERVABLE_H_
