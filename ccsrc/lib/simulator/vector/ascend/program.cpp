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
#include "simulator/vector/ascend/program.h"

#include <cmath>

#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "qubit_layout.h"
#include "simulator/circuit_tiling/builder.h"
#include "simulator/circuit_tiling/executor.h"
#include "simulator/circuit_tiling/planner.h"

namespace mindquantum::sim::ascend {
void ValidateResolver(const parameter::ParameterResolver& resolver) {
    auto scalar = [](const tensor::Tensor& value) {
        if (value.device != tensor::TDevice::CPU || value.dim != 1 || !value.data) {
            throw std::invalid_argument("Ascend parameters must be host scalars.");
        }
        const auto number = tensor::ops::cpu::to_vector<std::complex<double>>(value).front();
        if (!std::isfinite(number.real()) || !std::isfinite(number.imag()) || number.imag() != 0) {
            throw std::invalid_argument("Ascend parameters must be real and finite.");
        }
    };
    scalar(resolver.const_value);
    for (const auto& [name, value] : resolver.data_) {
        scalar(value);
    }
}

namespace ct = circuit_tiling;
namespace {
using complex_t = std::complex<double>;
using matrix_t = std::array<complex_t, 4>;

size_t Align(size_t count) {
    return (count + 7) & ~size_t{7};
}

float Coefficient(double value) {
    const auto result = static_cast<float>(value);
    if (!std::isfinite(result)) {
        throw std::invalid_argument("Ascend gate coefficients must be finite and representable in float32.");
    }
    return result;
}

// Tensor-product embedding: selected bits retain their order; every omitted
// bit labels an independent copy of the same small operation.
std::vector<uint64_t> Offsets(uint64_t mask) {
    std::vector<uint64_t> offsets{0};
    while (mask) {
        const auto bit = mask & (~mask + 1);
        const auto count = offsets.size();
        offsets.resize(2 * count);
        for (size_t i = 0; i < count; ++i) {
            offsets[count + i] = offsets[i] | bit;
        }
        mask ^= bit;
    }
    return offsets;
}

TileProgram MakeVectorTile(uint64_t mask, uint64_t plane) {
    TileProgram tile{};
    tile.size = uint32_t{1} << CountOne(mask);
    tile.tile_count = plane / tile.size;
    tile.external_mask = (plane - 1) & ~mask;
    tile.run = 1;
    while (mask & tile.run) {
        tile.run <<= 1;
    }
    tile.offsets = Offsets(mask & ~(uint64_t{tile.run} - 1));
    return tile;
}

uint64_t Project(uint64_t basis, uint64_t mask) {
    uint64_t result = 0, local = 1;
    while (mask) {
        const auto bit = mask & (~mask + 1);
        if (basis & bit) {
            result |= local;
        }
        local <<= 1;
        mask ^= bit;
    }
    return result;
}

std::optional<DenseProgram> DenseFusion(const ct::CircuitTile& source, const parameter::ParameterResolver& parameters,
                                        const ct::StateSpace& space, unsigned minimum_gates, QubitLayout& layout) {
    // Selection depends on operator support and work, never absolute qubit
    // positions or circuit identities. Short/diagonal regions stay on Vector.
    if (!minimum_gates || source.gates.size() < minimum_gates || space.dimension() < kDenseTileSize) {
        return std::nullopt;
    }
    uint64_t mask = 0;
    size_t mixers = 0;
    for (const auto& gate : source.gates) {
        const auto access = ct::DescribeGateAccess(gate, space).value();
        mask |= access.object_mask | access.control_mask;
        mixers += access.mixing_mask != 0;
    }
    if (CountOne(mask) > kMaxDenseQubits || mixers < minimum_gates) {
        return std::nullopt;
    }
    // Pad the operator with spectator identity factors, not reduced precision.
    for (qbit_t bit = 0; CountOne(mask) < 4; ++bit) {
        mask |= uint64_t{1} << bit;
    }
    const auto dimension = uint32_t{1} << CountOne(mask);
    auto tile = source;
    tile.qubits = ct::QubitsFromMask(mask);
    auto prepared = ct::Builder<complex_t>::Build(tile, parameters, space);
    // Builder projects operations to compact indices. Replace only traversal
    // to synthesize the small operator using the shared quantum arithmetic.
    prepared.layout.qubits = ct::QubitsFromMask(dimension - 1);
    prepared.layout.tile_count = 1;
    std::iota(prepared.layout.state_offsets.begin(), prepared.layout.state_offsets.end(), 0);
    for (const auto& operation : prepared.operations) {
        if (operation.external_control_mask != 0) {
            throw std::logic_error("A fixed dense operator cannot omit an external control.");
        }
    }
    DenseProgram result{};
    result.dimension = dimension;
    result.coefficients.resize(4 * dimension * dimension);
    std::vector<complex_t> column(dimension);
    // Synthesize a <=64x64 operator in double precision. This is compilation,
    // never CPU execution of the user's device-resident quantum state.
    for (uint32_t basis = 0; basis < dimension; ++basis) {
        std::fill(column.begin(), column.end(), complex_t{});
        column[basis] = 1;
        ct::ApplyTile(column.data(), prepared, 1);
        for (uint32_t row = 0; row < dimension; ++row) {
            const auto offset = row * 2 * dimension + basis;
            result.coefficients[offset] = Coefficient(column[row].real());
            result.coefficients[offset + dimension] = -Coefficient(column[row].imag());
            result.coefficients[offset + 2 * dimension * dimension] = Coefficient(column[row].imag());
            result.coefficients[offset + 2 * dimension * dimension + dimension] = Coefficient(column[row].real());
        }
    }
    const auto physical_support = layout.Physical(mask);
    auto tile_mask = physical_support | 7;
    for (qbit_t bit = 0; (uint64_t{1} << CountOne(tile_mask)) < kDenseTileSize; ++bit) {
        tile_mask |= uint64_t{1} << bit;
    }
    uint32_t run = 1;
    while (tile_mask & run) {
        run <<= 1;
    }
    result.packet = {space.dimension(), (space.dimension() - 1) & ~tile_mask, kDenseTileSize, run};
    result.offsets = Offsets(tile_mask & ~(uint64_t{run} - 1));
    const auto local_offsets = Offsets(tile_mask);
    const auto columns = kDenseTileSize / dimension;
    result.indices.resize(kDenseTileSize);
    for (uint32_t local = 0; local < kDenseTileSize; ++local) {
        const auto physical = local_offsets[local];
        const auto packed = Project(layout.Logical(physical), mask) * columns
                            + Project(physical, tile_mask & ~physical_support);
        result.indices[packed] = local * sizeof(float);
    }
    layout.AdoptPacked(tile_mask, mask);
    return result;
}

// Select an instruction form from the access pattern, not the quantum gate ID.
// Coalesced pairs use hardware strides; irregular pairs use indexed Gather.
std::vector<PairSpan> StridedPairs(const std::vector<size_t>& indices) {
    struct Run {
        size_t lhs, rhs, length;
    };
    std::vector<Run> runs;
    for (size_t i = 0; i < indices.size();) {
        const auto lhs = indices[i], rhs = indices[i + 1];
        size_t length = 1;
        while (i + 2 * length < indices.size() && indices[i + 2 * length] == lhs + length
               && indices[i + 2 * length + 1] == rhs + length) {
            ++length;
        }
        if (lhs % 8 || rhs % 8 || length % 8 || (length > 64 && length % 64)) {
            return {};
        }
        runs.push_back({lhs, rhs, length});
        i += 2 * length;
    }
    std::vector<PairSpan> spans;
    auto emit = [&](size_t lhs, size_t rhs, size_t mask, size_t count, size_t stride) {
        while (count) {
            const auto repeats = std::min(count, size_t{255});
            spans.push_back({static_cast<uint32_t>(lhs),
                             static_cast<uint32_t>(rhs),
                             static_cast<uint32_t>(mask),
                             static_cast<uint32_t>(repeats),
                             static_cast<uint32_t>(stride),
                             {}});
            count -= repeats;
            lhs += repeats * stride * 8;
            rhs += repeats * stride * 8;
        }
    };
    for (size_t i = 0; i < runs.size();) {
        const auto& first = runs[i];
        size_t end = i + 1;
        size_t step = 0;
        if (end < runs.size() && runs[end].lhs > first.lhs && runs[end].rhs > first.rhs) {
            step = runs[end].lhs - first.lhs;
            if (step % 8 == 0 && step / 8 <= 255) {
                while (end < runs.size() && runs[end].length == first.length
                       && runs[end].lhs == first.lhs + (end - i) * step
                       && runs[end].rhs == first.rhs + (end - i) * step) {
                    ++end;
                }
            }
        }
        if (first.length <= 64 || (end - i) > first.length / 64) {
            const auto mask = std::min(first.length, size_t{64});
            for (size_t offset = 0; offset < first.length; offset += mask) {
                emit(first.lhs + offset, first.rhs + offset, mask, end - i, end - i == 1 ? 8 : step / 8);
            }
        } else {
            for (size_t j = i; j < end; ++j) {
                emit(runs[j].lhs, runs[j].rhs, 64, runs[j].length / 64, 8);
            }
        }
        i = end;
    }
    // Many tiny disjoint spans lose SIMD occupancy and instruction throughput.
    return spans.size() <= 8 ? spans : std::vector<PairSpan>{};
}

// Cache compact access patterns, not fully expanded arrays or quantum gates.
// Its lifetime is exactly one compilation: no hidden cross-circuit cache and
// no retained references to gate objects or parameter values.
class Constants {
 public:
    explicit Constants(std::vector<uint32_t>& data) : data_(data) {
    }

    Instruction Permutation(std::vector<uint32_t> indices) {
        Instruction op{};
        op.kind = kPermutation;
        op.forward = Append(std::move(indices));
        return op;
    }

    Instruction Matrix(uint32_t size, uint32_t support, size_t dimension,
                       const std::vector<std::vector<size_t>>& compact) {
        auto key = std::make_tuple(size, support, dimension, compact);
        if (const auto found = matrices_.find(key); found != matrices_.end()) {
            return found->second;
        }
        const auto local = Offsets(support);
        const auto spectators = Offsets((size - 1) & ~support);
        std::vector<std::vector<size_t>> indices;
        for (const auto& block : compact) {
            auto& expanded = indices.emplace_back();
            expanded.reserve(block.size() * spectators.size());
            if (dimension == 2) {
                // A tile has at most 4096 bases and each pair owns a distinct
                // lhs. Canonicalize by direct addressing, not expanded sort.
                std::vector<size_t> partners(size, size);
                for (auto base : spectators) {
                    for (size_t i = 0; i < block.size(); i += 2) {
                        partners[base | local[block[i]]] = base | local[block[i + 1]];
                    }
                }
                for (size_t lhs = 0; lhs < size; ++lhs) {
                    if (partners[lhs] != size) {
                        expanded.push_back(lhs);
                        expanded.push_back(partners[lhs]);
                    }
                }
            } else {
                for (auto base : spectators) {
                    for (auto index : block) {
                        expanded.push_back(base | local[index]);
                    }
                }
            }
        }
        auto layout = MatrixLayout(size, dimension, indices);
        matrices_.emplace(std::move(key), layout);
        return layout;
    }

    uint32_t MatrixCoefficients(const std::vector<complex_t>& matrix) {
        std::vector<float> values;
        values.reserve(2 * matrix.size());
        for (const auto value : matrix) {
            values.push_back(Coefficient(value.real()));
            values.push_back(Coefficient(value.imag()));
        }
        std::vector<uint32_t> words(values.size());
        std::memcpy(words.data(), values.data(), values.size() * sizeof(float));
        return Append(std::move(words));
    }

    uint32_t Diagonal(uint32_t size, uint32_t support, const std::vector<complex_t>& factors) {
        std::vector<float> compact(2 * factors.size());
        for (size_t i = 0; i < factors.size(); ++i) {
            compact[i] = Coefficient(factors[i].real());
            compact[factors.size() + i] = Coefficient(factors[i].imag());
        }
        std::vector<uint32_t> words(compact.size());
        std::memcpy(words.data(), compact.data(), words.size() * sizeof(uint32_t));
        auto key = std::make_tuple(size, support, std::move(words));
        if (const auto found = diagonals_.find(key); found != diagonals_.end()) {
            return found->second;
        }
        std::vector<float> expanded(2 * size);
        const auto local = Offsets(support);
        for (auto base : Offsets((size - 1) & ~support)) {
            for (size_t i = 0; i < factors.size(); ++i) {
                expanded[base | local[i]] = compact[i];
                expanded[size + (base | local[i])] = compact[factors.size() + i];
            }
        }
        words.resize(expanded.size());
        std::memcpy(words.data(), expanded.data(), words.size() * sizeof(uint32_t));
        const auto offset = Append(std::move(words));
        diagonals_.emplace(std::move(key), offset);
        return offset;
    }

 private:
    uint32_t Append(std::vector<uint32_t> words) {
        words.resize(Align(words.size()), 0);
        if (data_.size() + words.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("Ascend program metadata exceeds its 32-bit word address space.");
        }
        const auto offset = static_cast<uint32_t>(data_.size());
        data_.insert(data_.end(), words.begin(), words.end());
        return offset;
    }

    Instruction MatrixLayout(uint32_t size, size_t dimension, const std::vector<std::vector<size_t>>& indices) {
        Instruction op{};
        op.kind = kMatrix;
        op.dimension = dimension;
        if (dimension == 2) {
            std::vector<PairSpan> spans;
            for (size_t block = 0; block < indices.size(); ++block) {
                const auto part = StridedPairs(indices[block]);
                if (part.empty()) {
                    spans.clear();
                    break;
                }
                op.block_groups[block] = part.size();
                spans.insert(spans.end(), part.begin(), part.end());
            }
            if (!spans.empty()) {
                std::vector<uint32_t> words(spans.size() * sizeof(PairSpan) / sizeof(uint32_t));
                std::memcpy(words.data(), spans.data(), words.size() * sizeof(uint32_t));
                op.kind = kStridedMatrix;
                op.groups = spans.size();
                op.forward = Append(std::move(words));
                return op;
            }
        }
        for (size_t block = 0; block < indices.size(); ++block) {
            op.block_groups[block] = Align(indices[block].size() / dimension);
            op.groups += op.block_groups[block];
        }
        std::vector<uint32_t> forward(dimension * op.groups, 0);
        std::vector<uint32_t> inverse(size, 0);
        std::vector<bool> used(size, false);
        size_t start = 0;
        for (size_t block = 0; block < indices.size(); ++block) {
            for (size_t column = 0; column < dimension; ++column) {
                for (size_t group = 0; group < indices[block].size() / dimension; ++group) {
                    const auto basis = indices[block][group * dimension + column];
                    const auto position = start + column * op.block_groups[block] + group;
                    forward[position] = basis * sizeof(float);
                    inverse[basis] = position * sizeof(float);
                    used[basis] = true;
                }
            }
            start += dimension * op.block_groups[block];
        }
        for (size_t basis = 0; basis < size; ++basis) {
            if (!used[basis]) {
                inverse[basis] = forward.size() * sizeof(float);
                forward.push_back(basis * sizeof(float));
            }
        }
        op.staging_size = Align(forward.size());
        if (op.staging_size > kStagingSize) {
            throw std::logic_error("Ascend matrix staging exceeds the UB contract.");
        }
        op.forward = Append(std::move(forward));
        op.inverse = Append(std::move(inverse));
        return op;
    }

    std::vector<uint32_t>& data_;
    std::map<std::tuple<uint32_t, uint32_t, size_t, std::vector<std::vector<size_t>>>, Instruction> matrices_;
    std::map<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>, uint32_t> diagonals_;
};

class Encoder {
 public:
    Encoder(uint64_t mask, uint64_t plane, Constants& constants, const QubitLayout& layout)
        : constants_(constants), layout_(layout), program_(MakeVectorTile(mask, plane)), mask_(mask) {
    }

    Encoder(const ct::CircuitTile& tile, const parameter::ParameterResolver& parameters, const ct::StateSpace& space,
            Constants& constants, const QubitLayout& layout)
        : Encoder(ct::MaskFromQubits(tile.qubits), space.dimension(), constants, layout) {
        for (const auto& gate : tile.gates) {
            // Let the shared Builder own every quantum-gate formula. Spectator
            // bits are omitted only during compilation, never during execution.
            const auto compact = ct::Builder<complex_t>::Build(Compact(gate, space), parameters, space);
            for (const auto& operation : compact.operations) {
                predicate_ = {};
                predicate_.controls = layout.Physical(operation.external_control_mask);
                std::visit([this](const auto& kernel) { Lower(kernel); }, operation.kernel);
            }
        }
        FlushDiagonal();
    }

    ct::CircuitTile Compact(const std::shared_ptr<BasicGate>& gate, const ct::StateSpace& space) {
        const auto access = ct::DescribeGateAccess(gate, space).value();
        const auto support = layout_.Logical(mask_) & (access.object_mask | access.control_mask);
        const auto physical_support = layout_.Physical(support);
        support_ = Project(physical_support, mask_);
        basis_.clear();
        for (auto basis : Offsets(support)) {
            basis_.push_back(Project(layout_.Physical(basis), physical_support));
        }
        return {ct::QubitsFromMask(support), {gate}};
    }

    // Explicit boundaries forbid diagonal folding across adjoint evaluation
    // points. Derivatives share the exact same semantic-kernel lowering.
    template <typename operation_t>
    std::pair<uint32_t, uint32_t> Encode(const operation_t& operation, bool derivative = false) {
        FlushDiagonal();
        const auto begin = Position();
        derivative_ = derivative;
        predicate_ = {};
        predicate_.controls = layout_.Physical(operation.external_control_mask);
        std::visit([this](const auto& kernel) { Lower(kernel); }, operation.kernel);
        FlushDiagonal();
        derivative_ = false;
        return {begin, Position() - begin};
    }

    uint32_t Position() const {
        if (program_.instructions.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("Ascend tile exceeds its 32-bit instruction address space.");
        }
        return static_cast<uint32_t>(program_.instructions.size());
    }

    TileProgram Finish() && {
        return std::move(program_);
    }

 private:
    void Matrix(const std::vector<std::vector<size_t>>& indices, size_t dimension,
                const std::vector<complex_t>& matrix) {
        if (indices.empty()) {
            return;
        }
        if (dimension > kMaxMatrixDimension) {
            throw std::invalid_argument("mqvector_npu supports dense custom matrices on at most six target qubits.");
        }
        if (indices.size() > kMaxMatrixBlocks || (dimension <= 4 && matrix.size() > 16)) {
            throw std::logic_error("Ascend matrix blocks exceed the instruction coefficient budget.");
        }
        FlushDiagonal();
        auto physical = indices;
        for (auto& block : physical) {
            for (auto& basis : block) {
                basis = basis_[basis];
            }
        }
        auto op = constants_.Matrix(program_.size, support_, dimension, physical);
        op.controls = predicate_.controls;
        op.parity_mask = predicate_.parity_mask;
        op.parity = predicate_.parity;
        if (derivative_) {
            op.kind |= kZeroUnmentioned;
        }
        if (dimension > 4) {
            op.coefficients = constants_.MatrixCoefficients(matrix);
        } else {
            for (size_t i = 0; i < matrix.size(); ++i) {
                op.matrix[2 * i] = Coefficient(matrix[i].real());
                op.matrix[2 * i + 1] = Coefficient(matrix[i].imag());
            }
        }
        program_.instructions.push_back(op);
    }

    template <typename pair_t>
    void Pairs(const std::vector<pair_t>& pairs, const matrix_t& matrix) {
        if (pairs.empty()) {
            return;
        }
        std::vector<size_t> indices;
        indices.reserve(2 * pairs.size());
        for (const auto& pair : pairs) {
            indices.push_back(pair.lhs);
            indices.push_back(pair.rhs);
        }
        Matrix({indices}, 2, {matrix.begin(), matrix.end()});
    }

    template <typename transform_t>
    void Scales(const std::vector<ct::Scale<complex_t>>& scales, const transform_t& transform) {
        if (scales.empty()) {
            return;
        }
        std::vector<complex_t> factors(size_t{1} << CountOne(support_), derivative_ ? complex_t{} : complex_t{1, 0});
        bool changed = derivative_;
        for (const auto& scale : scales) {
            const auto factor = transform(scale.factor);
            Coefficient(factor.real());
            Coefficient(factor.imag());
            factors[basis_[scale.basis]] = factor;
            changed |= factor != complex_t{1, 0};
        }
        if (!changed) {
            return;
        }
        if (diagonal_predicate_.controls != predicate_.controls
            || diagonal_predicate_.parity_mask != predicate_.parity_mask
            || diagonal_predicate_.parity != predicate_.parity) {
            FlushDiagonal();
        }
        if (diagonal_.empty()) {
            diagonal_ = std::move(factors);
            diagonal_support_ = support_;
            diagonal_predicate_ = predicate_;
            return;
        }
        // Fold adjacent diagonal operations in complex128 over just the union
        // of their supports; expand to the physical UB width only on emission.
        const auto combined_support = diagonal_support_ | support_;
        const auto offsets = Offsets(combined_support);
        std::vector<complex_t> combined(offsets.size());
        for (size_t i = 0; i < offsets.size(); ++i) {
            combined[i] = diagonal_[Project(offsets[i], diagonal_support_)] * factors[Project(offsets[i], support_)];
        }
        diagonal_ = std::move(combined);
        diagonal_support_ = combined_support;
    }

    void FlushDiagonal() {
        if (diagonal_.empty()) {
            return;
        }
        if (derivative_
            || std::any_of(diagonal_.begin(), diagonal_.end(), [](auto factor) { return factor != complex_t{1, 0}; })) {
            auto op = diagonal_predicate_;
            op.kind = kDiagonal | (derivative_ ? kZeroUnmentioned : 0);
            op.coefficients = constants_.Diagonal(program_.size, diagonal_support_, diagonal_);
            program_.instructions.push_back(op);
        }
        diagonal_.clear();
    }

    void Lower(const ct::HadamardKernel& kernel) {
        const auto c = std::sqrt(0.5);
        Pairs(kernel.pairs, {c, c, c, -c});
    }
    void Lower(const ct::PairMatrixKernel<complex_t>& kernel) {
        Pairs(kernel.pairs, kernel.matrix);
    }
    void Lower(const ct::SwapKernel& kernel) {
        Pairs(kernel.pairs, {0, 1, 1, 0});
    }
    void Lower(const ct::ISwapKernel<complex_t>& kernel) {
        Pairs(kernel.pairs, {0, kernel.phase, kernel.phase, 0});
    }
    void Lower(const ct::DiagonalKernel<complex_t>& kernel) {
        Scales(kernel.scales, [](auto value) { return value; });
    }
    void Lower(const ct::DenseMatrixKernel<complex_t>& kernel) {
        if (!kernel.indices.empty()) {
            Matrix({kernel.indices}, kernel.matrix_dim, kernel.matrix);
        }
    }
    void Lower(const ct::FSimKernel<complex_t>& kernel) {
        std::vector<ct::IndexPair> pairs;
        std::vector<ct::Scale<complex_t>> scales;
        for (const auto& item : kernel.indices) {
            pairs.push_back({item.lhs, item.rhs});
            scales.push_back({item.both, kernel.phase});
        }
        Pairs(pairs, kernel.exchange);
        Scales(scales, [](auto value) { return value; });
    }

    // Disjoint coefficient blocks share one state permutation. This consumes
    // the semantic IR's coefficients and indices, not quantum gate IDs.
    void PauliPairs(const std::vector<ct::PauliPair<complex_t>>& pairs, complex_t diagonal, complex_t multiplier) {
        using key_t = std::array<double, 4>;
        std::map<key_t, std::vector<ct::IndexPair>> groups;
        for (const auto& pair : pairs) {
            const auto a = multiplier * pair.lhs_from_rhs;
            const auto b = multiplier * pair.rhs_from_lhs;
            groups[{a.real(), a.imag(), b.real(), b.imag()}].push_back({pair.lhs, pair.rhs});
        }
        std::vector<std::vector<size_t>> blocks;
        std::vector<complex_t> matrices;
        for (auto& [coefficients, indices] : groups) {
            // The semantic builder enumerates Paulis in Gray-code order.
            // Independent pairs may be reordered into coalesced NPU accesses.
            std::sort(indices.begin(), indices.end(), [](const auto& a, const auto& b) { return a.lhs < b.lhs; });
            auto& block = blocks.emplace_back();
            block.reserve(2 * indices.size());
            for (const auto& pair : indices) {
                block.push_back(pair.lhs);
                block.push_back(pair.rhs);
            }
            matrices.insert(
                matrices.end(),
                {diagonal, {coefficients[0], coefficients[1]}, {coefficients[2], coefficients[3]}, diagonal});
        }
        Matrix(blocks, 2, matrices);
    }

    template <typename function_t>
    void WithParity(uint64_t mask, const function_t& emit) {
        predicate_.parity_mask = layout_.Physical(mask);
        for (uint32_t parity = 0; parity < (mask ? 2U : 1U); ++parity) {
            predicate_.parity = parity;
            emit(parity ? -1.0 : 1.0);
        }
    }

    void Lower(const ct::PauliKernel<complex_t>& kernel) {
        WithParity(kernel.external_z_mask, [&](double sign) {
            Scales(kernel.scales, [sign](auto value) { return sign * value; });
            PauliPairs(kernel.pairs, 0, sign);
        });
    }
    void Lower(const ct::PauliRotationKernel<complex_t>& kernel) {
        if (kernel.coefficients_prepared) {
            Scales(kernel.scales, [](auto value) { return value; });
            PauliPairs(kernel.pairs, kernel.cosine, 1);
        } else {
            WithParity(kernel.external_z_mask, [&](double sign) {
                const complex_t multiplier{0, -sign * kernel.sine};
                Scales(kernel.scales, [&](auto value) { return kernel.cosine + multiplier * value; });
                PauliPairs(kernel.pairs, kernel.cosine, multiplier);
            });
        }
    }
    void Lower(const ct::DepolarizingKernel<complex_t>&) {
        throw std::invalid_argument("mqvector_npu does not support density-matrix noise kernels.");
    }

    Constants& constants_;
    const QubitLayout& layout_;
    std::vector<size_t> basis_;
    TileProgram program_{};
    uint64_t mask_;
    bool derivative_ = false;
    Instruction predicate_{};
    uint32_t support_ = 0;
    Instruction diagonal_predicate_{};
    uint32_t diagonal_support_ = 0;
    std::vector<complex_t> diagonal_;
};

// Factor the final index permutation into bounded in-place UB permutations.
// This is a representation change, not an additional quantum operation.
void RestoreLayout(Program& program, Constants& constants, const QubitLayout& layout, uint64_t plane) {
    // Only Cube changes the layout, so restoration uses its packet budget,
    // independently of the caller's potentially smaller Vector tile width.
    const auto width = CountOne(kDenseTileSize - 1);
    std::vector<std::pair<qbit_t, qbit_t>> pending;
    uint64_t mask = 7;
    const auto flush = [&]() {
        if (pending.empty()) {
            return;
        }
        for (qbit_t bit = 0; CountOne(mask) < width; ++bit) {
            mask |= uint64_t{1} << bit;
        }
        auto tile = MakeVectorTile(mask, plane);
        const auto offsets = Offsets(mask);
        std::vector<uint32_t> indices(tile.size);
        for (uint32_t output = 0; output < tile.size; ++output) {
            auto input = offsets[output];
            for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
                const auto lhs = uint64_t{1} << it->first;
                const auto rhs = uint64_t{1} << it->second;
                if (bool(input & lhs) != bool(input & rhs)) {
                    input ^= lhs | rhs;
                }
            }
            indices[output] = Project(input, mask) * sizeof(float);
        }
        tile.instructions.push_back(constants.Permutation(std::move(indices)));
        program.tiles.emplace_back(std::move(tile));
        pending.clear();
        mask = 7;
    };
    for (const auto& swap : layout.Restoration()) {
        const auto support = (uint64_t{1} << swap.first) | (uint64_t{1} << swap.second);
        if (CountOne(mask | support) > width) {
            flush();
        }
        mask |= support;
        pending.push_back(swap);
    }
    flush();
}

uint32_t CheckedCount(size_t count) {
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("Ascend metadata exceeds its 32-bit index space.");
    }
    return static_cast<uint32_t>(count);
}

void ValidateCircuit(const std::vector<std::shared_ptr<BasicGate>>& circuit, qbit_t n_qubits) {
    if (n_qubits < 0 || n_qubits > 32) {
        throw std::invalid_argument("mqvector_npu requires 0 <= n_qubits <= 32, subject to available device memory.");
    }
    const auto logical_space = ct::StateSpace::Vector(n_qubits);
    for (const auto& gate : circuit) {
        if (!gate) {
            throw std::invalid_argument("Cannot compile a null gate.");
        }
        ct::detail::ValidateGateQubits(*gate, gate->GetObjQubits(), gate->GetCtrlQubits(), n_qubits);
        if (gate->GetID() != GateID::I && !ct::DescribeGateAccess(gate, logical_space)) {
            throw std::invalid_argument(fmt::format("mqvector_npu does not support gate {}.", gate->GetID()));
        }
        if (const auto* parameterized = dynamic_cast<const Parameterizable*>(gate.get())) {
            for (const auto& coefficient : parameterized->GetCoeffs()) {
                ValidateResolver(coefficient);
            }
        }
        // Capability validation precedes execution-policy selection: fusion
        // must not make an otherwise unsupported custom gate acceptable.
        if ((gate->GetID() == GateID::CUSTOM || gate->GetID() == GateID::CUSTOM_TWO_PARAM)
            && gate->GetObjQubits().size() > kMaxDenseQubits) {
            throw std::invalid_argument("mqvector_npu supports dense custom matrices on at most six target qubits.");
        }
    }
}

// BasicGate has no virtual Clone. These are only concrete copy constructors,
// not a second semantic gate compiler. PRs and tensor::Matrix deep-copy their
// CPU storage; the plan never retains mutable references to the source gates.
std::vector<std::shared_ptr<BasicGate>> Snapshot(const std::vector<std::shared_ptr<BasicGate>>& circuit) {
    std::vector<std::shared_ptr<BasicGate>> result;
    result.reserve(circuit.size());
    for (const auto& gate : circuit) {
#define MQ_ASCEND_COPY_GATE(id, type)                                                                                  \
    case GateID::id:                                                                                                   \
        result.push_back(std::make_shared<type>(*static_cast<const type*>(gate.get())));                               \
        break
        switch (gate->GetID()) {
            MQ_ASCEND_COPY_GATE(I, IGate);
            MQ_ASCEND_COPY_GATE(X, XGate);
            MQ_ASCEND_COPY_GATE(Y, YGate);
            MQ_ASCEND_COPY_GATE(Z, ZGate);
            MQ_ASCEND_COPY_GATE(H, HGate);
            MQ_ASCEND_COPY_GATE(S, SGate);
            MQ_ASCEND_COPY_GATE(Sdag, SdagGate);
            MQ_ASCEND_COPY_GATE(T, TGate);
            MQ_ASCEND_COPY_GATE(Tdag, TdagGate);
            MQ_ASCEND_COPY_GATE(SX, SXGate);
            MQ_ASCEND_COPY_GATE(SXdag, SXdagGate);
            MQ_ASCEND_COPY_GATE(SWAP, SWAPGate);
            MQ_ASCEND_COPY_GATE(ISWAP, ISWAPGate);
            MQ_ASCEND_COPY_GATE(RX, RXGate);
            MQ_ASCEND_COPY_GATE(RY, RYGate);
            MQ_ASCEND_COPY_GATE(RZ, RZGate);
            MQ_ASCEND_COPY_GATE(Rxx, RxxGate);
            MQ_ASCEND_COPY_GATE(Ryy, RyyGate);
            MQ_ASCEND_COPY_GATE(Rzz, RzzGate);
            MQ_ASCEND_COPY_GATE(Rxy, RxyGate);
            MQ_ASCEND_COPY_GATE(Rxz, RxzGate);
            MQ_ASCEND_COPY_GATE(Ryz, RyzGate);
            MQ_ASCEND_COPY_GATE(Givens, GivensGate);
            MQ_ASCEND_COPY_GATE(SWAPalpha, SWAPalphaGate);
            MQ_ASCEND_COPY_GATE(GP, GPGate);
            MQ_ASCEND_COPY_GATE(PS, PSGate);
            MQ_ASCEND_COPY_GATE(PauliString, PauliString);
            MQ_ASCEND_COPY_GATE(RPS, RotPauliString);
            MQ_ASCEND_COPY_GATE(U3, U3);
            MQ_ASCEND_COPY_GATE(FSim, FSim);
            MQ_ASCEND_COPY_GATE(Rn, Rn);
            MQ_ASCEND_COPY_GATE(CUSTOM, CustomGate);
            MQ_ASCEND_COPY_GATE(CUSTOM_TWO_PARAM, CustomTwoParamGate);
            default:
                throw std::invalid_argument("Cannot snapshot an unsupported Ascend gate.");
        }
#undef MQ_ASCEND_COPY_GATE
    }
    return result;
}

ct::CircuitTile AsTile(const ct::Schedule::value_type& entry, const ct::StateSpace& space) {
    if (const auto* grouped = std::get_if<ct::CircuitTile>(&entry)) {
        return *grouped;
    }
    auto gate = std::get<std::shared_ptr<BasicGate>>(entry);
    return {ct::QubitsFromMask(ct::DescribeGateAccess(gate, space).value().mixing_mask), {gate}};
}

uint64_t AlignedTileMask(uint64_t mask, unsigned width) {
    mask |= 7U;
    if (CountOne(mask) > width) {
        throw std::invalid_argument("Gate mixing/control width exceeds the Ascend DMA tile budget.");
    }
    for (qbit_t bit = 0; CountOne(mask) < width; ++bit) {
        mask |= uint64_t{1} << bit;
    }
    return mask;
}
}  // namespace

struct ForwardPlan::Impl {
    qbit_t n_qubits;
    unsigned tile_qubits;
    unsigned cube_min_gates;
    ct::Schedule schedule;
};

ForwardPlan::ForwardPlan(const std::vector<std::shared_ptr<BasicGate>>& circuit, qbit_t n_qubits, unsigned tile_qubits,
                         unsigned fusion_qubits, unsigned cube_min_gates) {
    ValidateCircuit(circuit, n_qubits);
    if (tile_qubits < 3 || tile_qubits > 12 || fusion_qubits < 1 || fusion_qubits > tile_qubits - 3) {
        throw std::invalid_argument("Require 3 <= tile_qubits <= 12 and 1 <= fusion_qubits <= tile_qubits - 3.");
    }
    const auto space = ct::StateSpace::Vector(n_qubits);
    const auto use_cube = cube_min_gates && space.dimension() >= kDenseTileSize;
    const auto capacity = use_cube ? ct::Planner::Capacity::FullSupport : ct::Planner::Capacity::Mixing;
    const auto budget = use_cube ? std::min(fusion_qubits, kMaxDenseQubits) : fusion_qubits;
    impl_ = std::make_shared<Impl>(
        Impl{n_qubits, tile_qubits, cube_min_gates, ct::Planner(space, budget, capacity).Plan(Snapshot(circuit))});
}

Program ForwardPlan::Bind(const parameter::ParameterResolver& parameters) const {
    ValidateResolver(parameters);
    const auto n_qubits = impl_->n_qubits;
    const auto logical_space = ct::StateSpace::Vector(n_qubits);
    const auto physical_space = ct::StateSpace::Vector(std::max(n_qubits, qbit_t{3}));
    const auto width = std::min(impl_->tile_qubits, static_cast<unsigned>(physical_space.state_qubits()));
    Program result{n_qubits, {}, {}};
    Constants constants(result.data);
    QubitLayout layout(physical_space.state_qubits());
    auto emit_vector = [&](ct::CircuitTile tile) {
        if (tile.gates.empty()) {
            return;
        }
        // Reserve at least three contiguous low bits for 32-byte FP32 DMA.
        const auto mask = AlignedTileMask(layout.Physical(ct::MaskFromQubits(tile.qubits)), width);
        tile.qubits = ct::QubitsFromMask(mask);
        auto encoded = Encoder(tile, parameters, physical_space, constants, layout).Finish();
        if (!encoded.instructions.empty()) {
            result.tiles.push_back(std::move(encoded));
        }
    };
    for (const auto& entry : impl_->schedule) {
        auto tile = AsTile(entry, logical_space);
        if (auto dense = DenseFusion(tile, parameters, logical_space, impl_->cube_min_gates, layout)) {
            result.tiles.emplace_back(std::move(*dense));
        } else {
            emit_vector(std::move(tile));
        }
    }
    RestoreLayout(result, constants, layout, physical_space.dimension());
    return result;
}

struct AdjointPlan::Impl {
    qbit_t n_qubits;
    MST<size_t> parameter_map;
    std::vector<ct::CircuitTile> tiles;
};

AdjointPlan::AdjointPlan(const std::vector<std::shared_ptr<BasicGate>>& circuit, const MST<size_t>& parameter_map,
                         qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits) {
    ValidateCircuit(circuit, n_qubits);
    if (tile_qubits < 3 || tile_qubits > 12 || fusion_qubits < 1 || fusion_qubits > 9) {
        throw std::invalid_argument("Require 3 <= adjoint tile_qubits <= 12 and 1 <= fusion_qubits <= 9.");
    }
    CheckedCount(parameter_map.size() + 1);
    std::vector<bool> used(parameter_map.size(), false);
    for (const auto& [name, column] : parameter_map) {
        if (column >= used.size() || used[column]) {
            throw std::invalid_argument("Adjoint parameter map must be a bijection onto [0, parameter_count).");
        }
        used[column] = true;
    }
    auto impl = std::make_shared<Impl>(Impl{n_qubits, parameter_map, {}});
    const auto space = ct::StateSpace::Vector(n_qubits);
    const auto width = std::min({tile_qubits, static_cast<unsigned>(CountOne(kMaxAdjointTileSize - 1)),
                                 static_cast<unsigned>(std::max(n_qubits, qbit_t{3}))});
    const auto budget = std::min(fusion_qubits, std::max(1U, width - 3));
    auto reversed = Snapshot(circuit);
    std::reverse(reversed.begin(), reversed.end());
    for (const auto& entry : ct::Planner(space, budget).PlanAdjoint(reversed)) {
        auto tile = AsTile(entry, space);
        tile.qubits = ct::QubitsFromMask(AlignedTileMask(ct::MaskFromQubits(tile.qubits), width));
        impl->tiles.push_back(std::move(tile));
    }
    // Even the empty circuit has an overlap, including for zero qubits.
    if (impl->tiles.empty()) {
        impl->tiles.push_back({ct::QubitsFromMask(AlignedTileMask(0, width)), {}});
    }
    impl_ = std::move(impl);
}

AdjointProgram AdjointPlan::Bind(const parameter::ParameterResolver& parameters) const {
    ValidateResolver(parameters);
    AdjointProgram result{impl_->n_qubits, CheckedCount(impl_->parameter_map.size() + 1), impl_->parameter_map, {}, {}};
    const auto space = ct::StateSpace::Vector(std::max(impl_->n_qubits, qbit_t{3}));
    const QubitLayout layout(space.state_qubits());  // Adjoint boundaries always use canonical layout.
    Constants constants(result.data);
    for (const auto& tile : impl_->tiles) {
        Encoder encoder(ct::MaskFromQubits(tile.qubits), space.dimension(), constants, layout);
        AdjointTileProgram encoded;
        for (const auto& gate : tile.gates) {
            const auto prepared = ct::Builder<complex_t>::BuildAdjoint(encoder.Compact(gate, space), parameters,
                                                                       impl_->parameter_map, space,
                                                                       ct::Builder<complex_t>::AdjointGates::Reversed);
            for (const auto& operation : prepared.operations) {
                if (operation.evolution.size() != 1 || operation.derivative_position != 0) {
                    throw std::invalid_argument("Ascend adjoint supports only single-image state-vector operations.");
                }
                const auto [begin, count] = encoder.Encode(operation.evolution.front());
                AdjointStep step{begin, count, CheckedCount(encoded.derivatives.size()),
                                 CheckedCount(operation.derivatives.size())};
                for (const auto& derivative : operation.derivatives) {
                    const auto [derivative_begin, derivative_count] = encoder.Encode(derivative, true);
                    encoded.derivatives.push_back({derivative_begin, derivative_count,
                                                   CheckedCount(encoded.targets.size()),
                                                   CheckedCount(derivative.targets.size())});
                    for (const auto& target : derivative.targets) {
                        encoded.targets.push_back({CheckedCount(target.output_index),
                                                   Coefficient(target.coefficient.real()),
                                                   Coefficient(target.coefficient.imag()), 0});
                    }
                }
                encoded.steps.push_back(step);
            }
        }
        CheckedCount(encoded.steps.size());
        CheckedCount(encoded.derivatives.size());
        CheckedCount(encoded.targets.size());
        static_cast<TileProgram&>(encoded) = std::move(encoder).Finish();
        result.tiles.push_back(std::move(encoded));
    }
    return result;
}

Program Compile(const std::vector<std::shared_ptr<BasicGate>>& circuit, const parameter::ParameterResolver& parameters,
                qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits, unsigned cube_min_gates) {
    return ForwardPlan(circuit, n_qubits, tile_qubits, fusion_qubits, cube_min_gates).Bind(parameters);
}

AdjointProgram CompileAdjoint(const std::vector<std::shared_ptr<BasicGate>>& circuit,
                              const parameter::ParameterResolver& parameters, const MST<size_t>& parameter_map,
                              qbit_t n_qubits, unsigned tile_qubits, unsigned fusion_qubits) {
    return AdjointPlan(circuit, parameter_map, n_qubits, tile_qubits, fusion_qubits).Bind(parameters);
}

Program CompileDerivative(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& parameters,
                          qbit_t n_qubits, unsigned tile_qubits) {
    ValidateCircuit({gate}, n_qubits);
    ValidateResolver(parameters);
    if (tile_qubits < 3 || tile_qubits > 12 || gate->GetID() == GateID::I) {
        throw std::invalid_argument("Invalid standalone Ascend derivative gate or tile width.");
    }
    // A constant CustomGate may be a fixed matrix, whose derivative wrapper
    // has no initialized callback. Its public API cannot distinguish that
    // constructor from a numerically bound custom callback gate. Require a
    // symbolic intrinsic parameter for this case rather than calling it.
    if (gate->GetID() == GateID::CUSTOM && !gate->Parameterized()) {
        throw std::invalid_argument("A standalone custom derivative requires a symbolic intrinsic parameter.");
    }
    const auto space = ct::StateSpace::Vector(std::max(n_qubits, qbit_t{3}));
    const auto access = ct::DescribeGateAccess(gate, space).value();
    const auto width = std::min(tile_qubits, static_cast<unsigned>(space.state_qubits()));
    // Include diagonal operands too: phase derivatives need zero on their
    // inactive operand sector. Controls outside the UB block are projected
    // separately below, so they do not constrain standalone derivative width.
    // Wide diagonal/Pauli strings may retain external parity instead. Builder
    // rejects any omitted operand that needs a local support projection.
    const auto operands = CountOne(access.object_mask | 7U) <= width ? access.object_mask : access.mixing_mask;
    const auto mask = AlignedTileMask(operands, width);
    auto local_gate = Snapshot({gate}).front();
    qbits_t local_controls;
    for (const auto qubit : gate->GetCtrlQubits()) {
        if (mask & (uint64_t{1} << qubit)) {
            local_controls.push_back(qubit);
        }
    }
    // Copy only placement metadata; keep the concrete gate, coefficients and
    // custom callbacks intact. Builder still owns the intrinsic derivative.
    static_cast<BasicGate&>(*local_gate) = BasicGate(gate->GetID(), gate->GetObjQubits(), local_controls);
    Program result{n_qubits, {}, {}};
    Constants constants(result.data);
    const QubitLayout layout(space.state_qubits());
    Encoder encoder(mask, space.dimension(), constants, layout);
    const auto prepared = ct::Builder<complex_t>::BuildDerivative(encoder.Compact(local_gate, space), parameters,
                                                                  space);
    for (const auto& operation : prepared.operations) {
        encoder.Encode(operation);
    }
    auto tile = std::move(encoder).Finish();
    auto external_controls = access.control_mask & ~mask;
    if (external_controls) {
        Instruction zero{};
        zero.kind = kDiagonal;
        zero.coefficients = constants.Diagonal(tile.size, 0, {complex_t{}});
        while (external_controls) {
            const auto bit = external_controls & (~external_controls + 1);
            // A one-bit parity predicate selects precisely the inactive sector.
            zero.parity_mask = bit;
            tile.instructions.push_back(zero);
            external_controls ^= bit;
        }
    }
    result.tiles.emplace_back(std::move(tile));
    return result;
}
}  // namespace mindquantum::sim::ascend
