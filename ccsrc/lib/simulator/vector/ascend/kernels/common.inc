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
#ifndef MINDQUANTUM_ASCEND_COMMON_INC_
#define MINDQUANTUM_ASCEND_COMMON_INC_

#include "kernel_operator.h"
#include "simulator/vector/ascend/device_program.h"

namespace mindquantum::sim::ascend::device {
using namespace AscendC;  // NOLINT

struct Complex {
    float real;
    float imag;
};

template <HardEvent event, event_t local = EVENT_ID0>
__aicore__ inline void Sync() {
    // Fetch does NOT allocate. Avoid the MTE3_S event reserved by CANN's
    // Matmul client; Set/Wait consumes this temporary fence before returning.
    const auto id = event == HardEvent::MTE3_S ? static_cast<event_t>(GetTPipePtr()->FetchEventID(event)) : local;
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

__aicore__ inline uint64_t Deposit(uint64_t value, uint64_t mask) {
    uint64_t result = 0;
    while (mask) {
        const auto bit = mask & (~mask + 1);
        if (value & 1) {
            result |= bit;
        }
        value >>= 1;
        mask ^= bit;
    }
    return result;
}

__aicore__ inline uint32_t Parity(uint64_t value) {
    value ^= value >> 32;
    value ^= value >> 16;
    value ^= value >> 8;
    value ^= value >> 4;
    return (0x6996U >> (value & 15U)) & 1U;
}

struct CompensatedSum {
    float sum = 0.0F;
    float error = 0.0F;
    __aicore__ inline void AddValue(float value) {
        const float adjusted = value - error;
        const float next = sum + adjusted;
        error = (next - sum) - adjusted;
        sum = next;
    }
};

// AIV-owned records need no atomics, tags, eviction policy or parameter limit.
// Callers aggregate spatial contractions before updating a record.
class PartialSums {
 public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR partials, uint32_t output_size) {
        output_size_ = output_size;
        partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partials));
        pipe.InitBuffer(transfer_buf_, kGradientPartialStride * sizeof(float));
    }

    __aicore__ inline void Add(uint32_t output, Complex value, Complex correction = {0.0F, 0.0F}) {
        const auto offset = (static_cast<uint64_t>(GetBlockIdx()) * output_size_ + output) * kGradientPartialStride;
        auto record = transfer_buf_.Get<float>();
        DataCopy(record, partials_[offset], kGradientPartialStride);
        Sync<HardEvent::MTE2_S, EVENT_ID1>();
        const float contributions[] = {value.real, value.imag};
        const float corrections[] = {correction.real, correction.imag};
        for (uint32_t i = 0; i < 2; ++i) {
            CompensatedSum sum{record.GetValue(i), record.GetValue(i + 2)};
            sum.AddValue(contributions[i]);
            sum.AddValue(corrections[i]);
            record.SetValue(i, sum.sum);
            record.SetValue(i + 2, sum.error);
        }
        Sync<HardEvent::S_MTE3, EVENT_ID1>();
        DataCopy(partials_[offset], record, kGradientPartialStride);
        Sync<HardEvent::MTE3_MTE2, EVENT_ID1>();
    }

 private:
    TBuf<TPosition::VECCALC> transfer_buf_;
    GlobalTensor<float> partials_;
    uint32_t output_size_;
};

template <uint32_t Capacity>
__aicore__ inline void ComplexProducts(LocalTensor<float> work, LocalTensor<float> bra, LocalTensor<float> ket,
                                       uint32_t size) {
    Mul(work, bra, ket, size);
    Mul(work[Capacity], bra[Capacity], ket[Capacity], size);
    Mul(work[2 * Capacity], bra, ket[Capacity], size);
    Mul(work[3 * Capacity], bra[Capacity], ket, size);
    PipeBarrier<PIPE_V>();
    Add(work, work, work[Capacity], size);
    Sub(work[2 * Capacity], work[2 * Capacity], work[3 * Capacity], size);
    PipeBarrier<PIPE_V>();
}

// Balanced reduction shared by Vector and Cube adjoints. Four product planes
// allow the real and imaginary reductions to advance together.
template <uint32_t Capacity>
class InnerProduct {
 public:
    __aicore__ inline void Init(TPipe& pipe, uint32_t size) {
        size_ = size;
        pipe.InitBuffer(work_buf_, 4 * Capacity * sizeof(float));
    }

    __aicore__ inline Complex Dot(LocalTensor<float> bra, LocalTensor<float> ket) {
        auto work = work_buf_.Get<float>();
        ComplexProducts<Capacity>(work, bra, ket, size_);
        for (uint32_t half = size_ / 2; half >= 8; half /= 2) {
            Add(work, work, work[half], half);
            Add(work[2 * Capacity], work[2 * Capacity], work[2 * Capacity + half], half);
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_S>();
        const Complex result{SumEight(work), SumEight(work[2 * Capacity])};
        Sync<HardEvent::S_V>();
        return result;
    }

 private:
    __aicore__ inline float SumEight(LocalTensor<float> values) {
        const float a = values.GetValue(0) + values.GetValue(1);
        const float b = values.GetValue(2) + values.GetValue(3);
        const float c = values.GetValue(4) + values.GetValue(5);
        const float d = values.GetValue(6) + values.GetValue(7);
        return (a + b) + (c + d);
    }

    TBuf<TPosition::VECCALC> work_buf_;
    uint32_t size_;
};
}  // namespace mindquantum::sim::ascend::device
#endif  // MINDQUANTUM_ASCEND_COMMON_INC_
