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
#ifndef MINDQUANTUM_ASCEND_TILE_EXECUTOR_INC_
#define MINDQUANTUM_ASCEND_TILE_EXECUTOR_INC_

#include "common.inc"

namespace mindquantum::sim::ascend::device {
using namespace AscendC;  // NOLINT

__aicore__ inline bool Matches(const __gm__ Instruction& op, uint64_t base) {
    const bool matches = (base & op.controls) == op.controls && Parity(base & op.parity_mask) == op.parity;
    return matches != bool(op.kind & kInvertPredicate);
}

// Shared Vector quantum instruction executor. The caller owns the resident
// SoA tile (plane stride Capacity); only bounded reusable scratch lives here.
// Derivative pieces differ ONLY in their zero extension on untouched bases.
template <uint32_t Capacity>
class TileExecutor {
 public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR data, uint32_t size) {
        data_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(data));
        coefficients_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(data));
        size_ = size;
        pipe.InitBuffer(input_buf_, 2 * Staging * sizeof(float));
        pipe.InitBuffer(output_buf_, 2 * Staging * sizeof(float));
        pipe.InitBuffer(index_buf_, (Staging + Capacity) * sizeof(uint32_t));
        pipe.InitBuffer(temp_buf_, Capacity * sizeof(float));
    }

    // Consecutive SoA states share the instruction's resident metadata. Keep
    // one interpreter body instead of expanding a separate paired executor.
    __aicore__ inline void Apply(const __gm__ Instruction& op, LocalTensor<float> state, uint32_t count = 1) {
        Prepare(op);
        for (uint32_t i = 0; i < count; ++i) {
            ApplyLoaded(op, state[2 * Capacity * i]);
        }
    }

 private:
    static constexpr uint32_t Staging = Capacity + kMatrixPadding;

    __aicore__ inline void ApplyLoaded(const __gm__ Instruction& op, LocalTensor<float> state) {
        state_ = state;
        const auto kind = op.kind & ~kInstructionFlags;
        if (kind == kMatrix) {
            Matrix(op);
        } else if (kind == kStridedMatrix) {
            StridedMatrix(op);
        } else if (kind == kPermutation) {
            Permute(op);
        } else {
            Diagonal(op);
        }
    }

    __aicore__ inline void Prepare(const __gm__ Instruction& op) {
        const auto kind = op.kind & ~kInstructionFlags;
        if (kind == kStridedMatrix) {
            return;
        }
        // Neither state's arithmetic overwrites this instruction's metadata.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (kind == kMatrix) {
            DataCopy(index_buf_.Get<uint32_t>(), data_[op.forward], op.staging_size);
            DataCopy(index_buf_.Get<uint32_t>()[Staging], data_[op.inverse], size_);
        } else if (kind == kPermutation) {
            DataCopy(index_buf_.Get<uint32_t>(), data_[op.forward], size_);
        } else {
            auto factors = input_buf_.Get<float>();
            DataCopy(factors, coefficients_[op.coefficients], size_);
            DataCopy(factors[Staging], coefficients_[op.coefficients + size_], size_);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    template <typename... args_t>
    __aicore__ inline void Accumulate(LocalTensor<float> dst, LocalTensor<float> src, float coefficient,
                                      bool initialized, args_t... args) {
        if (initialized) {
            Axpy(dst, src, coefficient, args...);
        } else {
            Muls(dst, src, coefficient, args...);
        }
    }

    template <typename... args_t>
    __aicore__ inline void AccumulateComplex(LocalTensor<float> dst_real, LocalTensor<float> dst_imag,
                                             LocalTensor<float> src_real, LocalTensor<float> src_imag, float real,
                                             float imag, bool& initialized, args_t... args) {
        // A purely imaginary first contribution still initializes both planes.
        if (real != 0.0F) {
            Accumulate(dst_real, src_real, real, initialized, args...);
            Accumulate(dst_imag, src_imag, real, initialized, args...);
            initialized = true;
            PipeBarrier<PIPE_V>();
        }
        if (imag != 0.0F) {
            Accumulate(dst_real, src_imag, -imag, initialized, args...);
            Accumulate(dst_imag, src_real, imag, initialized, args...);
            initialized = true;
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Matrix(const __gm__ Instruction& op) {
        auto state = state_;
        auto input = input_buf_.Get<float>();
        auto output = output_buf_.Get<float>();
        auto index = index_buf_.Get<uint32_t>();
        const uint32_t staging_size = op.staging_size;
        Gather(input, state, index, 0, staging_size);
        Gather(input[Staging], state[Capacity], index, 0, staging_size);
        PipeBarrier<PIPE_V>();
        const auto active = op.dimension * op.groups;
        if (active < staging_size) {
            if (op.kind & kZeroUnmentioned) {
                Duplicate(output[active], 0.0F, staging_size - active);
                Duplicate(output[Staging + active], 0.0F, staging_size - active);
            } else {
                Adds(output[active], input[active], 0.0F, staging_size - active);
                Adds(output[Staging + active], input[Staging + active], 0.0F, staging_size - active);
            }
            PipeBarrier<PIPE_V>();
        }
        uint32_t start = 0;
        for (uint32_t block = 0; block < kMaxMatrixBlocks && op.block_groups[block]; ++block) {
            const auto groups = op.block_groups[block];
            const auto matrix = block * op.dimension * op.dimension;
            for (uint32_t row = 0; row < op.dimension; ++row) {
                const auto dst = start + row * groups;
                bool initialized = false;
                for (uint32_t column = 0; column < op.dimension; ++column) {
                    const auto src = start + column * groups;
                    const auto coefficient = 2 * (matrix + row * op.dimension + column);
                    const float real = op.dimension > 4 ? coefficients_.GetValue(op.coefficients + coefficient)
                                                        : op.matrix[coefficient];
                    const float imag = op.dimension > 4 ? coefficients_.GetValue(op.coefficients + coefficient + 1)
                                                        : op.matrix[coefficient + 1];
                    AccumulateComplex(output[dst], output[Staging + dst], input[src], input[Staging + src], real, imag,
                                      initialized, groups);
                }
                if (!initialized) {
                    Duplicate(output[dst], 0.0F, groups);
                    Duplicate(output[Staging + dst], 0.0F, groups);
                    PipeBarrier<PIPE_V>();
                }
            }
            start += op.dimension * groups;
        }
        Gather(state, output, index[Staging], 0, size_);
        Gather(state[Capacity], output[Staging], index[Staging], 0, size_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StridedMatrix(const __gm__ Instruction& op) {
        auto state = state_;
        auto output = output_buf_.Get<float>();
        auto spans = reinterpret_cast<__gm__ PairSpan*>(data_.GetPhyAddr(op.forward));
        const bool derivative = (op.kind & kZeroUnmentioned) != 0;
        if (derivative) {
            Duplicate(output, 0.0F, size_);
            Duplicate(output[Staging], 0.0F, size_);
            PipeBarrier<PIPE_V>();
        }
        uint32_t block = 0, end = op.block_groups[0];
        for (uint32_t i = 0; i < op.groups; ++i) {
            if (i == end) {
                end += op.block_groups[++block];
            }
            const uint32_t lhs = spans[i].lhs, rhs = spans[i].rhs;
            const uint64_t mask = spans[i].mask;
            const uint8_t repeats = spans[i].repeats, stride = spans[i].stride;
            const UnaryRepeatParams strides{1, 1, stride, stride};
            for (uint32_t row = 0; row < 2; ++row) {
                const auto dst = row ? rhs : lhs;
                bool initialized = false;
                for (uint32_t column = 0; column < 2; ++column) {
                    const auto src = column ? rhs : lhs;
                    const float real = op.matrix[2 * (block * 4 + row * 2 + column)];
                    const float imag = op.matrix[2 * (block * 4 + row * 2 + column) + 1];
                    AccumulateComplex(output[dst], output[Staging + dst], state[src], state[Capacity + src], real, imag,
                                      initialized, mask, repeats, strides);
                }
                if (!initialized) {
                    Duplicate(output[dst], 0.0F, mask, repeats, 1, stride);
                    Duplicate(output[Staging + dst], 0.0F, mask, repeats, 1, stride);
                    PipeBarrier<PIPE_V>();
                }
            }
            // Both rows finish before either source row is overwritten.
            if (!derivative) {
                Adds(state[lhs], output[lhs], 0.0F, mask, repeats, strides);
                Adds(state[rhs], output[rhs], 0.0F, mask, repeats, strides);
                Adds(state[Capacity + lhs], output[Staging + lhs], 0.0F, mask, repeats, strides);
                Adds(state[Capacity + rhs], output[Staging + rhs], 0.0F, mask, repeats, strides);
                PipeBarrier<PIPE_V>();
            }
        }
        if (derivative) {
            // The zero tail must also reach the destination, without ever
            // destroying source rows needed by a later span.
            Adds(state, output, 0.0F, size_);
            Adds(state[Capacity], output[Staging], 0.0F, size_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Permute(const __gm__ Instruction& op) {
        auto state = state_;
        auto input = input_buf_.Get<float>();
        auto index = index_buf_.Get<uint32_t>();
        DataCopy(input, state, size_);
        DataCopy(input[Staging], state[Capacity], size_);
        PipeBarrier<PIPE_V>();
        Gather(state, input, index, 0, size_);
        Gather(state[Capacity], input[Staging], index, 0, size_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Diagonal(const __gm__ Instruction& op) {
        auto state = state_;
        auto factors = input_buf_.Get<float>();
        auto output = output_buf_.Get<float>();
        auto temp = temp_buf_.Get<float>();
        // Derivative diagonals are already zero-extended by the compiler.
        Mul(temp, state[Capacity], factors[Staging], size_);
        Mul(output, state, factors[Staging], size_);
        PipeBarrier<PIPE_V>();
        Mul(state, state, factors, size_);
        Mul(state[Capacity], state[Capacity], factors, size_);
        PipeBarrier<PIPE_V>();
        Sub(state, state, temp, size_);
        Add(state[Capacity], state[Capacity], output, size_);
        PipeBarrier<PIPE_V>();
    }

    TBuf<TPosition::VECCALC> input_buf_, output_buf_, index_buf_, temp_buf_;
    GlobalTensor<float> coefficients_;
    GlobalTensor<uint32_t> data_;
    LocalTensor<float> state_;
    uint32_t size_;
};
}  // namespace mindquantum::sim::ascend::device
#endif  // MINDQUANTUM_ASCEND_TILE_EXECUTOR_INC_
