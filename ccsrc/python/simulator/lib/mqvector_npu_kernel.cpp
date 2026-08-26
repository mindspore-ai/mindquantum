/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
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

#include "kernel_operator.h"

using AscendC::Barrier;
using AscendC::CacheLine;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DcciDst;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetSubBlockIdx;
using AscendC::GetTaskRation;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::WriteGmByPassDCache;
using AscendC::get_block_idx;

// The state is laid out as interleaved complex64 values: [real0, imag0, ...].
// Each block owns a disjoint collection of target-qubit amplitude pairs.
struct CircuitGate {
    uint64_t target_mask;
    uint64_t control_mask;
    uint32_t kind;
    uint32_t operation;
    float m00_real;
    float m00_imag;
    float m01_real;
    float m01_imag;
    float m10_real;
    float m10_imag;
    float m11_real;
    float m11_imag;
    float cosine;
    float sine;
};

static_assert(sizeof(CircuitGate) == 64, "mqvector_npu circuit gate layout must stay ABI-compatible with the host.");

constexpr uint32_t kMatrixGeneral = 0U;
constexpr uint32_t kMatrixX = 1U;
constexpr uint32_t kMatrixY = 2U;
constexpr uint32_t kMatrixZ = 3U;
constexpr uint32_t kMatrixH = 4U;
constexpr uint32_t kMatrixRX = 5U;
constexpr uint32_t kMatrixRY = 6U;
constexpr uint32_t kMatrixRZ = 7U;
constexpr uint32_t kMatrixPS = 8U;

__aicore__ inline void ProbeWrite(__gm__ uint32_t* address, uint32_t value) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    WriteGmByPassDCache(address, value);
#else
    *address = value;
#endif
}

__aicore__ inline uint32_t FloatToBits(float value) {
    union {
        float value;
        uint32_t bits;
    } cast;
    cast.value = value;
    return cast.bits;
}

__aicore__ inline uint32_t DecodeLocalMeta(float value) {
    return FloatToBits(value) & 0x007FFFFFU;
}

__aicore__ inline float ReadState(__gm__ float* address) {
    return *address;
}

__aicore__ inline void WriteState(__gm__ float* address, float value) {
    *address = value;
}

__aicore__ inline void FlushStateCacheLine(GlobalTensor<float>& state_gm, uint64_t offset) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && defined(MQVECTOR_NPU_ENABLE_STATE_CACHE_FLUSH)
    constexpr uint64_t kFloatsPerCacheLine = 16U;
    const uint64_t aligned_offset = offset & ~(kFloatsPerCacheLine - 1U);
    GlobalTensor<int64_t> cache_line;
    cache_line.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(state_gm.GetPhyAddr(aligned_offset)));
    Barrier();
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(cache_line);
    Barrier();
#endif
}

__aicore__ inline uint64_t ScatterBits(uint64_t value, uint64_t mask) {
    uint64_t result = 0U;
    uint64_t source_bit = 1U;
    while (mask != 0U) {
        const uint64_t target_bit = mask & (~mask + 1U);
        if ((value & source_bit) != 0U) {
            result |= target_bit;
        }
        source_bit <<= 1U;
        mask ^= target_bit;
    }
    return result;
}

__aicore__ inline uint32_t PopCountLocal(uint64_t value) {
    uint32_t count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

__aicore__ inline void ApplyMatrixPair(
    GlobalTensor<float>& state_gm, __gm__ const CircuitGate* gate, uint64_t zero_offset, uint64_t one_offset) {
    const uint32_t operation = gate->operation;

    if (operation == kMatrixZ) {
        const float one_real = ReadState(state_gm.GetPhyAddr(one_offset));
        const float one_imag = ReadState(state_gm.GetPhyAddr(one_offset + 1U));
        WriteState(state_gm.GetPhyAddr(one_offset), -one_real);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), -one_imag);
        return;
    }
    if (operation == kMatrixPS) {
        const float cosine = gate->m11_real;
        const float sine = gate->m11_imag;
        const float one_real = ReadState(state_gm.GetPhyAddr(one_offset));
        const float one_imag = ReadState(state_gm.GetPhyAddr(one_offset + 1U));
        WriteState(state_gm.GetPhyAddr(one_offset), cosine * one_real - sine * one_imag);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), cosine * one_imag + sine * one_real);
        return;
    }

    const float zero_real = ReadState(state_gm.GetPhyAddr(zero_offset));
    const float zero_imag = ReadState(state_gm.GetPhyAddr(zero_offset + 1U));
    const float one_real = ReadState(state_gm.GetPhyAddr(one_offset));
    const float one_imag = ReadState(state_gm.GetPhyAddr(one_offset + 1U));

    if (operation == kMatrixX) {
        WriteState(state_gm.GetPhyAddr(zero_offset), one_real);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), one_imag);
        WriteState(state_gm.GetPhyAddr(one_offset), zero_real);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), zero_imag);
        return;
    }
    if (operation == kMatrixY) {
        WriteState(state_gm.GetPhyAddr(zero_offset), one_imag);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), -one_real);
        WriteState(state_gm.GetPhyAddr(one_offset), -zero_imag);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), zero_real);
        return;
    }
    if (operation == kMatrixH) {
        const float scale = gate->m00_real;
        WriteState(state_gm.GetPhyAddr(zero_offset), scale * (zero_real + one_real));
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), scale * (zero_imag + one_imag));
        WriteState(state_gm.GetPhyAddr(one_offset), scale * (zero_real - one_real));
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), scale * (zero_imag - one_imag));
        return;
    }
    if (operation == kMatrixRX) {
        const float cosine = gate->m00_real;
        const float sine = -gate->m01_imag;
        WriteState(state_gm.GetPhyAddr(zero_offset), cosine * zero_real + sine * one_imag);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), cosine * zero_imag - sine * one_real);
        WriteState(state_gm.GetPhyAddr(one_offset), sine * zero_imag + cosine * one_real);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), -sine * zero_real + cosine * one_imag);
        return;
    }
    if (operation == kMatrixRY) {
        const float cosine = gate->m00_real;
        const float sine = gate->m10_real;
        WriteState(state_gm.GetPhyAddr(zero_offset), cosine * zero_real - sine * one_real);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), cosine * zero_imag - sine * one_imag);
        WriteState(state_gm.GetPhyAddr(one_offset), sine * zero_real + cosine * one_real);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), sine * zero_imag + cosine * one_imag);
        return;
    }
    if (operation == kMatrixRZ) {
        const float cosine = gate->m00_real;
        const float sine = -gate->m00_imag;
        WriteState(state_gm.GetPhyAddr(zero_offset), cosine * zero_real + sine * zero_imag);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), cosine * zero_imag - sine * zero_real);
        WriteState(state_gm.GetPhyAddr(one_offset), cosine * one_real - sine * one_imag);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), cosine * one_imag + sine * one_real);
        return;
    }

    const float out_zero_real = gate->m00_real * zero_real - gate->m00_imag * zero_imag
                                + gate->m01_real * one_real - gate->m01_imag * one_imag;
    const float out_zero_imag = gate->m00_real * zero_imag + gate->m00_imag * zero_real
                                + gate->m01_real * one_imag + gate->m01_imag * one_real;
    const float out_one_real = gate->m10_real * zero_real - gate->m10_imag * zero_imag
                               + gate->m11_real * one_real - gate->m11_imag * one_imag;
    const float out_one_imag = gate->m10_real * zero_imag + gate->m10_imag * zero_real
                               + gate->m11_real * one_imag + gate->m11_imag * one_real;

    WriteState(state_gm.GetPhyAddr(zero_offset), out_zero_real);
    WriteState(state_gm.GetPhyAddr(zero_offset + 1U), out_zero_imag);
    WriteState(state_gm.GetPhyAddr(one_offset), out_one_real);
    WriteState(state_gm.GetPhyAddr(one_offset + 1U), out_one_imag);
}

__aicore__ inline float ReadLocal(__ubuf__ float* state, uint64_t offset) {
    return state[offset];
}

__aicore__ inline void WriteLocal(__ubuf__ float* state, uint64_t offset, float value) {
    state[offset] = value;
}

__aicore__ inline void ApplyMatrixPairLocal(
    __ubuf__ float* state, __gm__ const CircuitGate* gate, uint64_t zero_offset, uint64_t one_offset) {
    const uint32_t operation = gate->operation;

    if (operation == kMatrixZ) {
        const float one_real = ReadLocal(state, one_offset);
        const float one_imag = ReadLocal(state, one_offset + 1U);
        WriteLocal(state, one_offset, -one_real);
        WriteLocal(state, one_offset + 1U, -one_imag);
        return;
    }
    if (operation == kMatrixPS) {
        const float cosine = gate->m11_real;
        const float sine = gate->m11_imag;
        const float one_real = ReadLocal(state, one_offset);
        const float one_imag = ReadLocal(state, one_offset + 1U);
        WriteLocal(state, one_offset, cosine * one_real - sine * one_imag);
        WriteLocal(state, one_offset + 1U, cosine * one_imag + sine * one_real);
        return;
    }

    const float zero_real = ReadLocal(state, zero_offset);
    const float zero_imag = ReadLocal(state, zero_offset + 1U);
    const float one_real = ReadLocal(state, one_offset);
    const float one_imag = ReadLocal(state, one_offset + 1U);

    if (operation == kMatrixX) {
        WriteLocal(state, zero_offset, one_real);
        WriteLocal(state, zero_offset + 1U, one_imag);
        WriteLocal(state, one_offset, zero_real);
        WriteLocal(state, one_offset + 1U, zero_imag);
        return;
    }
    if (operation == kMatrixY) {
        WriteLocal(state, zero_offset, one_imag);
        WriteLocal(state, zero_offset + 1U, -one_real);
        WriteLocal(state, one_offset, -zero_imag);
        WriteLocal(state, one_offset + 1U, zero_real);
        return;
    }
    if (operation == kMatrixH) {
        const float scale = gate->m00_real;
        WriteLocal(state, zero_offset, scale * (zero_real + one_real));
        WriteLocal(state, zero_offset + 1U, scale * (zero_imag + one_imag));
        WriteLocal(state, one_offset, scale * (zero_real - one_real));
        WriteLocal(state, one_offset + 1U, scale * (zero_imag - one_imag));
        return;
    }
    if (operation == kMatrixRX) {
        const float cosine = gate->m00_real;
        const float sine = -gate->m01_imag;
        WriteLocal(state, zero_offset, cosine * zero_real + sine * one_imag);
        WriteLocal(state, zero_offset + 1U, cosine * zero_imag - sine * one_real);
        WriteLocal(state, one_offset, sine * zero_imag + cosine * one_real);
        WriteLocal(state, one_offset + 1U, -sine * zero_real + cosine * one_imag);
        return;
    }
    if (operation == kMatrixRY) {
        const float cosine = gate->m00_real;
        const float sine = gate->m10_real;
        WriteLocal(state, zero_offset, cosine * zero_real - sine * one_real);
        WriteLocal(state, zero_offset + 1U, cosine * zero_imag - sine * one_imag);
        WriteLocal(state, one_offset, sine * zero_real + cosine * one_real);
        WriteLocal(state, one_offset + 1U, sine * zero_imag + cosine * one_imag);
        return;
    }
    if (operation == kMatrixRZ) {
        const float cosine = gate->m00_real;
        const float sine = -gate->m00_imag;
        WriteLocal(state, zero_offset, cosine * zero_real + sine * zero_imag);
        WriteLocal(state, zero_offset + 1U, cosine * zero_imag - sine * zero_real);
        WriteLocal(state, one_offset, cosine * one_real - sine * one_imag);
        WriteLocal(state, one_offset + 1U, cosine * one_imag + sine * one_real);
        return;
    }

    const float out_zero_real = gate->m00_real * zero_real - gate->m00_imag * zero_imag
                                + gate->m01_real * one_real - gate->m01_imag * one_imag;
    const float out_zero_imag = gate->m00_real * zero_imag + gate->m00_imag * zero_real
                                + gate->m01_real * one_imag + gate->m01_imag * one_real;
    const float out_one_real = gate->m10_real * zero_real - gate->m10_imag * zero_imag
                               + gate->m11_real * one_real - gate->m11_imag * one_imag;
    const float out_one_imag = gate->m10_real * zero_imag + gate->m10_imag * zero_real
                               + gate->m11_real * one_imag + gate->m11_imag * one_real;

    WriteLocal(state, zero_offset, out_zero_real);
    WriteLocal(state, zero_offset + 1U, out_zero_imag);
    WriteLocal(state, one_offset, out_one_real);
    WriteLocal(state, one_offset + 1U, out_one_imag);
}

// Report the logical block coordinates observed by the generated Ascend C
// launch stub. This tiny diagnostic kernel is kept separate from state
// evolution so multi-block scheduling can be checked before it is enabled for
// quantum gates.
extern "C" __global__ __aicore__ void mqvector_probe_block_dim(GM_ADDR output, uint32_t output_size) {
    GlobalTensor<uint32_t> output_gm;
    // Keep each block record on its own 64B cache line. Otherwise concurrent
    // scalar writes from AIV blocks can overwrite neighboring records.
    constexpr uint32_t kRecordWidth = 16;
    output_gm.SetGlobalBuffer((__gm__ uint32_t*)output, output_size * kRecordWidth);

    const uint64_t block_idx = static_cast<uint64_t>(GetBlockIdx());
    if (block_idx < output_size) {
        const uint64_t record_offset = block_idx * kRecordWidth;
        ProbeWrite(output_gm.GetPhyAddr(record_offset), static_cast<uint32_t>(GetBlockIdx()));
        ProbeWrite(output_gm.GetPhyAddr(record_offset + 1U), static_cast<uint32_t>(get_block_idx()));
        ProbeWrite(output_gm.GetPhyAddr(record_offset + 2U), static_cast<uint32_t>(GetSubBlockIdx()));
        ProbeWrite(output_gm.GetPhyAddr(record_offset + 3U), static_cast<uint32_t>(GetTaskRation()));
        ProbeWrite(output_gm.GetPhyAddr(record_offset + 4U), static_cast<uint32_t>(GetBlockNum()));
    }
}

// Reset the state vector in device memory. Keeping reset on the device avoids
// materializing and copying a full 2^n-element host vector before every run.
extern "C" __global__ __aicore__ void mqvector_reset_state(GM_ADDR state, uint64_t dim) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);

    // On the AIV-only launch, only the even physical block ids are
    // assigned work. Compress them into consecutive logical workers.
    const uint64_t physical_block_idx = static_cast<uint64_t>(GetBlockIdx());
    if ((physical_block_idx & 1U) != 0U) {
        return;
    }
    const uint64_t block_idx = physical_block_idx >> 1U;
    const uint64_t block_num = (static_cast<uint64_t>(GetBlockNum()) + 1U) >> 1U;
    const uint64_t work_begin = (dim * 2U * block_idx) / block_num;
    const uint64_t work_end = (dim * 2U * (block_idx + 1U)) / block_num;
    for (uint64_t index = work_begin; index < work_end; ++index) {
        if (index == 0U) {
            continue;
        }
        WriteState(state_gm.GetPhyAddr(index), 0.0F);
    }
    if (block_idx == 0U) {
        WriteState(state_gm.GetPhyAddr(0U), 1.0F);
    }
}

extern "C" __global__ __aicore__ void mqvector_apply_matrix(
    GM_ADDR state, uint64_t dim, uint64_t target_mask, uint64_t control_mask, float m00_real, float m00_imag,
    float m01_real, float m01_imag, float m10_real, float m10_imag, float m11_real, float m11_imag) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);

    const uint64_t pair_count = dim >> 1U;
    const uint64_t physical_block_idx = static_cast<uint64_t>(GetBlockIdx());
    if ((physical_block_idx & 1U) != 0U) {
        return;
    }
    const uint64_t block_idx = physical_block_idx >> 1U;
    const uint64_t block_num = (static_cast<uint64_t>(GetBlockNum()) + 1U) >> 1U;
    const uint64_t target_low_mask = target_mask - 1U;
    // Keep all pairs touching the same 64B GM cache-line group in one block.
    // For low target bits, four pairs fill one line; for higher target bits,
    // eight pairs share the two separated lines touched by the gate.
    const uint64_t group_width = target_mask < 8U ? 4U : 8U;
    const uint64_t group_count = (pair_count + group_width - 1U) / group_width;
    const uint64_t group_begin = (group_count * block_idx) / block_num;
    const uint64_t group_end = (group_count * (block_idx + 1U)) / block_num;
    const uint64_t work_begin = group_begin * group_width;
    const uint64_t work_end = (group_end * group_width < pair_count) ? group_end * group_width : pair_count;

    for (uint64_t pair_index = work_begin; pair_index < work_end; ++pair_index) {
        const uint64_t low_bits = pair_index & target_low_mask;
        // pair_index omits the target bit. Bits above it must be shifted
        // left when reconstructing the basis index with target=0.
        const uint64_t high_bits = (pair_index & ~(target_mask - 1U)) << 1U;
        const uint64_t basis_zero = high_bits | low_bits;
        if ((basis_zero & control_mask) != control_mask) {
            continue;
        }

        const uint64_t basis_one = basis_zero | target_mask;
        const uint64_t zero_offset = basis_zero << 1U;
        const uint64_t one_offset = basis_one << 1U;

        const float zero_real = ReadState(state_gm.GetPhyAddr(zero_offset));
        const float zero_imag = ReadState(state_gm.GetPhyAddr(zero_offset + 1U));
        const float one_real = ReadState(state_gm.GetPhyAddr(one_offset));
        const float one_imag = ReadState(state_gm.GetPhyAddr(one_offset + 1U));

        const float out_zero_real = m00_real * zero_real - m00_imag * zero_imag + m01_real * one_real
                                    - m01_imag * one_imag;
        const float out_zero_imag = m00_real * zero_imag + m00_imag * zero_real + m01_real * one_imag
                                    + m01_imag * one_real;
        const float out_one_real = m10_real * zero_real - m10_imag * zero_imag + m11_real * one_real
                                   - m11_imag * one_imag;
        const float out_one_imag = m10_real * zero_imag + m10_imag * zero_real + m11_real * one_imag
                                   + m11_imag * one_real;

        WriteState(state_gm.GetPhyAddr(zero_offset), out_zero_real);
        WriteState(state_gm.GetPhyAddr(zero_offset + 1U), out_zero_imag);
        WriteState(state_gm.GetPhyAddr(one_offset), out_one_real);
        WriteState(state_gm.GetPhyAddr(one_offset + 1U), out_one_imag);
        FlushStateCacheLine(state_gm, zero_offset);
        if ((zero_offset & ~15ULL) != (one_offset & ~15ULL)) {
            FlushStateCacheLine(state_gm, one_offset);
        }
    }
}

// Apply the double-qubit gates generated by mindquantum.utils.random_circuit.
// Operations are 0=SWAP, 1=Rxx, 2=Ryy, and 3=Rzz. The target-bit order is
// irrelevant for this symmetric gate set, so the lower/higher target bits are
// used only to form each disjoint amplitude quartet.
extern "C" __global__ __aicore__ void mqvector_apply_two_qubit(
    GM_ADDR state, uint64_t dim, uint64_t target_mask, uint64_t control_mask, uint32_t operation, float cosine,
    float sine) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);

    const uint64_t pair_count = dim >> 2U;
    const uint64_t physical_block_idx = static_cast<uint64_t>(GetBlockIdx());
    if ((physical_block_idx & 1U) != 0U) {
        return;
    }
    const uint64_t block_idx = physical_block_idx >> 1U;
    const uint64_t block_num = (static_cast<uint64_t>(GetBlockNum()) + 1U) >> 1U;

    const uint64_t target_low = target_mask & (~target_mask + 1U);
    const uint64_t target_high = target_mask ^ target_low;
    uint64_t low_pos = 0;
    uint64_t high_pos = 0;
    for (uint64_t bit = target_low; bit > 1U; bit >>= 1U) {
        ++low_pos;
    }
    for (uint64_t bit = target_high; bit > 1U; bit >>= 1U) {
        ++high_pos;
    }

    const uint64_t low_count_mask = target_low - 1U;
    const uint64_t middle_width = high_pos - low_pos - 1U;
    const uint64_t middle_count_mask = (static_cast<uint64_t>(1U) << middle_width) - 1U;
    const uint64_t high_shift = high_pos - 1U;
    // A quartet group owns one complete set of cache lines for its four
    // amplitudes. Two target bits below the 8-amplitude cache line leave two
    // zero-basis quartets per line; one leaves four; otherwise eight.
    const uint64_t target_bits_below_cache_line = (low_pos < 3U ? 1U : 0U) + (high_pos < 3U ? 1U : 0U);
    const uint64_t group_width = 8U >> target_bits_below_cache_line;
    const uint64_t group_count = (pair_count + group_width - 1U) / group_width;
    const uint64_t group_begin = (group_count * block_idx) / block_num;
    const uint64_t group_end = (group_count * (block_idx + 1U)) / block_num;
    const uint64_t work_begin = group_begin * group_width;
    const uint64_t work_end = (group_end * group_width < pair_count) ? group_end * group_width : pair_count;

    for (uint64_t pair_index = work_begin; pair_index < work_end; ++pair_index) {
        const uint64_t low_bits = pair_index & low_count_mask;
        const uint64_t middle_bits = (pair_index >> low_pos) & middle_count_mask;
        const uint64_t high_bits = pair_index >> high_shift;
        const uint64_t basis_zero = low_bits | (middle_bits << (low_pos + 1U)) | (high_bits << (high_pos + 1U));
        if ((basis_zero & control_mask) != control_mask) {
            continue;
        }

        const uint64_t offset0 = basis_zero << 1U;
        const uint64_t offset1 = (basis_zero | target_low) << 1U;
        const uint64_t offset2 = (basis_zero | target_high) << 1U;
        const uint64_t offset3 = (basis_zero | target_mask) << 1U;

        if (operation == 0U) {
            const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
            const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
            const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
            const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
            WriteState(state_gm.GetPhyAddr(offset1), two_real);
            WriteState(state_gm.GetPhyAddr(offset1 + 1U), two_imag);
            WriteState(state_gm.GetPhyAddr(offset2), one_real);
            WriteState(state_gm.GetPhyAddr(offset2 + 1U), one_imag);
            FlushStateCacheLine(state_gm, offset1);
            if ((offset1 & ~15ULL) != (offset2 & ~15ULL)) {
                FlushStateCacheLine(state_gm, offset2);
            }
            continue;
        }

        const float zero_real = ReadState(state_gm.GetPhyAddr(offset0));
        const float zero_imag = ReadState(state_gm.GetPhyAddr(offset0 + 1U));
        const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
        const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
        const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
        const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
        const float three_real = ReadState(state_gm.GetPhyAddr(offset3));
        const float three_imag = ReadState(state_gm.GetPhyAddr(offset3 + 1U));

        float out_zero_real;
        float out_zero_imag;
        float out_one_real;
        float out_one_imag;
        float out_two_real;
        float out_two_imag;
        float out_three_real;
        float out_three_imag;

        if (operation == 1U) {
            // Rxx(theta) = cos(theta/2) I - i sin(theta/2) X tensor X.
            out_zero_real = cosine * zero_real + sine * three_imag;
            out_zero_imag = cosine * zero_imag - sine * three_real;
            out_one_real = cosine * one_real + sine * two_imag;
            out_one_imag = cosine * one_imag - sine * two_real;
            out_two_real = cosine * two_real + sine * one_imag;
            out_two_imag = cosine * two_imag - sine * one_real;
            out_three_real = cosine * three_real + sine * zero_imag;
            out_three_imag = cosine * three_imag - sine * zero_real;
        } else if (operation == 2U) {
            // Ryy(theta) = cos(theta/2) I - i sin(theta/2) Y tensor Y.
            out_zero_real = cosine * zero_real - sine * three_imag;
            out_zero_imag = cosine * zero_imag + sine * three_real;
            out_one_real = cosine * one_real + sine * two_imag;
            out_one_imag = cosine * one_imag - sine * two_real;
            out_two_real = cosine * two_real + sine * one_imag;
            out_two_imag = cosine * two_imag - sine * one_real;
            out_three_real = cosine * three_real - sine * zero_imag;
            out_three_imag = cosine * three_imag + sine * zero_real;
        } else {
            // Rzz(theta) applies exp(-i theta/2) to |00>, |11> and
            // exp(+i theta/2) to |01>, |10>.
            out_zero_real = cosine * zero_real + sine * zero_imag;
            out_zero_imag = cosine * zero_imag - sine * zero_real;
            out_one_real = cosine * one_real - sine * one_imag;
            out_one_imag = cosine * one_imag + sine * one_real;
            out_two_real = cosine * two_real - sine * two_imag;
            out_two_imag = cosine * two_imag + sine * two_real;
            out_three_real = cosine * three_real + sine * three_imag;
            out_three_imag = cosine * three_imag - sine * three_real;
        }

        WriteState(state_gm.GetPhyAddr(offset0), out_zero_real);
        WriteState(state_gm.GetPhyAddr(offset0 + 1U), out_zero_imag);
        WriteState(state_gm.GetPhyAddr(offset1), out_one_real);
        WriteState(state_gm.GetPhyAddr(offset1 + 1U), out_one_imag);
        WriteState(state_gm.GetPhyAddr(offset2), out_two_real);
        WriteState(state_gm.GetPhyAddr(offset2 + 1U), out_two_imag);
        WriteState(state_gm.GetPhyAddr(offset3), out_three_real);
        WriteState(state_gm.GetPhyAddr(offset3 + 1U), out_three_imag);
        FlushStateCacheLine(state_gm, offset0);
        if ((offset0 & ~15ULL) != (offset1 & ~15ULL)) {
            FlushStateCacheLine(state_gm, offset1);
        }
        if ((offset0 & ~15ULL) != (offset2 & ~15ULL) && (offset1 & ~15ULL) != (offset2 & ~15ULL)) {
            FlushStateCacheLine(state_gm, offset2);
        }
        if ((offset0 & ~15ULL) != (offset3 & ~15ULL) && (offset1 & ~15ULL) != (offset3 & ~15ULL)
            && (offset2 & ~15ULL) != (offset3 & ~15ULL)) {
            FlushStateCacheLine(state_gm, offset3);
        }
    }
}

// Execute a complete circuit in one already validated block. The current
// single-block scheduling is intentional: each gate depends on the state
// written by the preceding gate, so this fused path avoids global
// synchronization while removing one launch per gate.
extern "C" __global__ __aicore__ void mqvector_apply_circuit(
    GM_ADDR state, GM_ADDR gates, uint64_t dim, uint32_t gate_count) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);
    __gm__ const CircuitGate* gates_gm = reinterpret_cast<__gm__ const CircuitGate*>(gates);

    const uint64_t block_idx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t block_num = static_cast<uint64_t>(GetBlockNum());

    for (uint32_t gate_index = 0; gate_index < gate_count; ++gate_index) {
        __gm__ const CircuitGate* gate = gates_gm + gate_index;
        if (gate->kind == 0U) {
            const uint64_t pair_count = dim >> 1U;
            const uint64_t target_low_mask = gate->target_mask - 1U;
            for (uint64_t pair_index = block_idx; pair_index < pair_count; pair_index += block_num) {
                const uint64_t low_bits = pair_index & target_low_mask;
                const uint64_t high_bits = (pair_index & ~(gate->target_mask - 1U)) << 1U;
                const uint64_t basis_zero = high_bits | low_bits;
                if ((basis_zero & gate->control_mask) != gate->control_mask) {
                    continue;
                }

                const uint64_t basis_one = basis_zero | gate->target_mask;
                const uint64_t zero_offset = basis_zero << 1U;
                const uint64_t one_offset = basis_one << 1U;
                ApplyMatrixPair(state_gm, gate, zero_offset, one_offset);
            }
            continue;
        }

        const uint64_t quartet_count = dim >> 2U;
        const uint64_t target_low = gate->target_mask & (~gate->target_mask + 1U);
        const uint64_t target_high = gate->target_mask ^ target_low;
        uint64_t low_pos = 0;
        uint64_t high_pos = 0;
        for (uint64_t bit = target_low; bit > 1U; bit >>= 1U) {
            ++low_pos;
        }
        for (uint64_t bit = target_high; bit > 1U; bit >>= 1U) {
            ++high_pos;
        }

        const uint64_t low_count_mask = target_low - 1U;
        const uint64_t middle_width = high_pos - low_pos - 1U;
        const uint64_t middle_count_mask = (static_cast<uint64_t>(1U) << middle_width) - 1U;
        const uint64_t high_shift = high_pos - 1U;

        for (uint64_t quartet_index = block_idx; quartet_index < quartet_count; quartet_index += block_num) {
            const uint64_t low_bits = quartet_index & low_count_mask;
            const uint64_t middle_bits = (quartet_index >> low_pos) & middle_count_mask;
            const uint64_t high_bits = quartet_index >> high_shift;
            const uint64_t basis_zero =
                low_bits | (middle_bits << (low_pos + 1U)) | (high_bits << (high_pos + 1U));
            if ((basis_zero & gate->control_mask) != gate->control_mask) {
                continue;
            }

            const uint64_t offset0 = basis_zero << 1U;
            const uint64_t offset1 = (basis_zero | target_low) << 1U;
            const uint64_t offset2 = (basis_zero | target_high) << 1U;
            const uint64_t offset3 = (basis_zero | gate->target_mask) << 1U;

            if (gate->operation == 0U) {
                const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
                const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
                const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
                const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
                WriteState(state_gm.GetPhyAddr(offset1), two_real);
                WriteState(state_gm.GetPhyAddr(offset1 + 1U), two_imag);
                WriteState(state_gm.GetPhyAddr(offset2), one_real);
                WriteState(state_gm.GetPhyAddr(offset2 + 1U), one_imag);
                continue;
            }

            const float zero_real = ReadState(state_gm.GetPhyAddr(offset0));
            const float zero_imag = ReadState(state_gm.GetPhyAddr(offset0 + 1U));
            const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
            const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
            const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
            const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
            const float three_real = ReadState(state_gm.GetPhyAddr(offset3));
            const float three_imag = ReadState(state_gm.GetPhyAddr(offset3 + 1U));

            float out_zero_real;
            float out_zero_imag;
            float out_one_real;
            float out_one_imag;
            float out_two_real;
            float out_two_imag;
            float out_three_real;
            float out_three_imag;

            if (gate->operation == 1U) {
                out_zero_real = gate->cosine * zero_real + gate->sine * three_imag;
                out_zero_imag = gate->cosine * zero_imag - gate->sine * three_real;
                out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                out_three_real = gate->cosine * three_real + gate->sine * zero_imag;
                out_three_imag = gate->cosine * three_imag - gate->sine * zero_real;
            } else if (gate->operation == 2U) {
                out_zero_real = gate->cosine * zero_real - gate->sine * three_imag;
                out_zero_imag = gate->cosine * zero_imag + gate->sine * three_real;
                out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                out_three_real = gate->cosine * three_real - gate->sine * zero_imag;
                out_three_imag = gate->cosine * three_imag + gate->sine * zero_real;
            } else {
                out_zero_real = gate->cosine * zero_real + gate->sine * zero_imag;
                out_zero_imag = gate->cosine * zero_imag - gate->sine * zero_real;
                out_one_real = gate->cosine * one_real - gate->sine * one_imag;
                out_one_imag = gate->cosine * one_imag + gate->sine * one_real;
                out_two_real = gate->cosine * two_real - gate->sine * two_imag;
                out_two_imag = gate->cosine * two_imag + gate->sine * two_real;
                out_three_real = gate->cosine * three_real + gate->sine * three_imag;
                out_three_imag = gate->cosine * three_imag - gate->sine * three_real;
            }

            WriteState(state_gm.GetPhyAddr(offset0), out_zero_real);
            WriteState(state_gm.GetPhyAddr(offset0 + 1U), out_zero_imag);
            WriteState(state_gm.GetPhyAddr(offset1), out_one_real);
            WriteState(state_gm.GetPhyAddr(offset1 + 1U), out_one_imag);
            WriteState(state_gm.GetPhyAddr(offset2), out_two_real);
            WriteState(state_gm.GetPhyAddr(offset2 + 1U), out_two_imag);
            WriteState(state_gm.GetPhyAddr(offset3), out_three_real);
            WriteState(state_gm.GetPhyAddr(offset3 + 1U), out_three_imag);
        }
    }
}

// Execute a circuit segment whose target qubits are contained in local_mask.
// Each logical worker owns one independent tile over the remaining qubits, so
// the whole segment can run with block-level parallelism without requiring a
// cross-block barrier between gates.
extern "C" __global__ __aicore__ void mqvector_apply_local_circuit(
    GM_ADDR state, GM_ADDR gates, GM_ADDR local_indices, uint64_t dim, uint32_t gate_count, uint64_t local_mask,
    uint32_t local_qubits) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);
    __gm__ const CircuitGate* gates_gm = reinterpret_cast<__gm__ const CircuitGate*>(gates);
    __gm__ const uint32_t* local_indices_gm = reinterpret_cast<__gm__ const uint32_t*>(local_indices);

    const uint64_t physical_block_idx = static_cast<uint64_t>(GetBlockIdx());
    if ((physical_block_idx & 1U) != 0U) {
        return;
    }

    const uint64_t block_idx = physical_block_idx >> 1U;
    const uint64_t block_num = (static_cast<uint64_t>(GetBlockNum()) + 1U) >> 1U;
    const uint64_t local_dim = static_cast<uint64_t>(1U) << local_qubits;
    if (local_dim == 0U || local_dim > dim) {
        return;
    }
    const uint64_t tile_count = dim >> local_qubits;
    const uint64_t all_state_mask = dim - 1U;
    const uint64_t nonlocal_mask = all_state_mask & ~local_mask;
    const uint64_t local_index_mask = (local_dim - 1U);

    for (uint64_t tile_index = block_idx; tile_index < tile_count; tile_index += block_num) {
        const uint64_t tile_prefix = ScatterBits(tile_index, nonlocal_mask);
        for (uint32_t gate_index = 0; gate_index < gate_count; ++gate_index) {
            __gm__ const CircuitGate* gate = gates_gm + gate_index;
            const uint64_t local_control_mask = gate->control_mask & local_mask;
            const uint64_t nonlocal_control_mask = gate->control_mask & nonlocal_mask;
            if ((tile_prefix & nonlocal_control_mask) != nonlocal_control_mask) {
                continue;
            }
            if (gate->kind == 0U) {
                const uint64_t pair_count = local_dim >> 1U;
                const uint64_t target_bit = gate->target_mask & (~gate->target_mask + 1U);
                const uint64_t target_local_pos = static_cast<uint64_t>(
                    PopCountLocal(local_mask & (target_bit - 1U)));
                const uint64_t lower_local_mask = (uint64_t{1} << target_local_pos) - 1U;
                for (uint64_t pair_index = 0; pair_index < pair_count; ++pair_index) {
                    const uint64_t local_index =
                        (pair_index & lower_local_mask)
                        | ((pair_index & ~lower_local_mask) << 1U);
                    const uint64_t basis_zero =
                        tile_prefix | static_cast<uint64_t>(local_indices_gm[local_index & local_index_mask]);
                    if ((basis_zero & local_control_mask) != local_control_mask) {
                        continue;
                    }

                    const uint64_t basis_one = basis_zero | gate->target_mask;
                    const uint64_t zero_offset = basis_zero << 1U;
                    const uint64_t one_offset = basis_one << 1U;
                    ApplyMatrixPair(state_gm, gate, zero_offset, one_offset);
                    FlushStateCacheLine(state_gm, zero_offset);
                    if ((zero_offset & ~15ULL) != (one_offset & ~15ULL)) {
                        FlushStateCacheLine(state_gm, one_offset);
                    }
                }
                continue;
            }

            const uint64_t quartet_count = local_dim >> 2U;
            const uint64_t target_low = gate->target_mask & (~gate->target_mask + 1U);
            const uint64_t target_high = gate->target_mask ^ target_low;
            const uint64_t low_local_pos =
                static_cast<uint64_t>(PopCountLocal(local_mask & (target_low - 1U)));
            const uint64_t high_local_pos =
                static_cast<uint64_t>(PopCountLocal(local_mask & (target_high - 1U)));
            const uint64_t low_local_mask = (uint64_t{1} << low_local_pos) - 1U;
            const uint64_t middle_local_width = high_local_pos - low_local_pos - 1U;
            const uint64_t middle_local_mask =
                (uint64_t{1} << middle_local_width) - 1U;

            for (uint64_t quartet_index = 0; quartet_index < quartet_count; ++quartet_index) {
                const uint64_t low_bits = quartet_index & low_local_mask;
                const uint64_t middle_bits =
                    (quartet_index >> low_local_pos) & middle_local_mask;
                const uint64_t high_bits = quartet_index >> (high_local_pos - 1U);
                const uint64_t local_index =
                    low_bits | (middle_bits << (low_local_pos + 1U))
                    | (high_bits << (high_local_pos + 1U));
                const uint64_t basis_zero =
                    tile_prefix | static_cast<uint64_t>(local_indices_gm[local_index & local_index_mask]);
                if ((basis_zero & local_control_mask) != local_control_mask) {
                    continue;
                }

                const uint64_t offset0 = basis_zero << 1U;
                const uint64_t offset1 = (basis_zero | target_low) << 1U;
                const uint64_t offset2 = (basis_zero | target_high) << 1U;
                const uint64_t offset3 = (basis_zero | gate->target_mask) << 1U;

                if (gate->operation == 0U) {
                    const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
                    const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
                    const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
                    const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
                    WriteState(state_gm.GetPhyAddr(offset1), two_real);
                    WriteState(state_gm.GetPhyAddr(offset1 + 1U), two_imag);
                    WriteState(state_gm.GetPhyAddr(offset2), one_real);
                    WriteState(state_gm.GetPhyAddr(offset2 + 1U), one_imag);
                    FlushStateCacheLine(state_gm, offset1);
                    if ((offset1 & ~15ULL) != (offset2 & ~15ULL)) {
                        FlushStateCacheLine(state_gm, offset2);
                    }
                    continue;
                }

                const float zero_real = ReadState(state_gm.GetPhyAddr(offset0));
                const float zero_imag = ReadState(state_gm.GetPhyAddr(offset0 + 1U));
                const float one_real = ReadState(state_gm.GetPhyAddr(offset1));
                const float one_imag = ReadState(state_gm.GetPhyAddr(offset1 + 1U));
                const float two_real = ReadState(state_gm.GetPhyAddr(offset2));
                const float two_imag = ReadState(state_gm.GetPhyAddr(offset2 + 1U));
                const float three_real = ReadState(state_gm.GetPhyAddr(offset3));
                const float three_imag = ReadState(state_gm.GetPhyAddr(offset3 + 1U));

                float out_zero_real;
                float out_zero_imag;
                float out_one_real;
                float out_one_imag;
                float out_two_real;
                float out_two_imag;
                float out_three_real;
                float out_three_imag;

                if (gate->operation == 1U) {
                    out_zero_real = gate->cosine * zero_real + gate->sine * three_imag;
                    out_zero_imag = gate->cosine * zero_imag - gate->sine * three_real;
                    out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                    out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                    out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                    out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                    out_three_real = gate->cosine * three_real + gate->sine * zero_imag;
                    out_three_imag = gate->cosine * three_imag - gate->sine * zero_real;
                } else if (gate->operation == 2U) {
                    out_zero_real = gate->cosine * zero_real - gate->sine * three_imag;
                    out_zero_imag = gate->cosine * zero_imag + gate->sine * three_real;
                    out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                    out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                    out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                    out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                    out_three_real = gate->cosine * three_real - gate->sine * zero_imag;
                    out_three_imag = gate->cosine * three_imag + gate->sine * zero_real;
                } else {
                    out_zero_real = gate->cosine * zero_real + gate->sine * zero_imag;
                    out_zero_imag = gate->cosine * zero_imag - gate->sine * zero_real;
                    out_one_real = gate->cosine * one_real - gate->sine * one_imag;
                    out_one_imag = gate->cosine * one_imag + gate->sine * one_real;
                    out_two_real = gate->cosine * two_real - gate->sine * two_imag;
                    out_two_imag = gate->cosine * two_imag + gate->sine * two_real;
                    out_three_real = gate->cosine * three_real + gate->sine * three_imag;
                    out_three_imag = gate->cosine * three_imag - gate->sine * three_real;
                }

                WriteState(state_gm.GetPhyAddr(offset0), out_zero_real);
                WriteState(state_gm.GetPhyAddr(offset0 + 1U), out_zero_imag);
                WriteState(state_gm.GetPhyAddr(offset1), out_one_real);
                WriteState(state_gm.GetPhyAddr(offset1 + 1U), out_one_imag);
                WriteState(state_gm.GetPhyAddr(offset2), out_two_real);
                WriteState(state_gm.GetPhyAddr(offset2 + 1U), out_two_imag);
                WriteState(state_gm.GetPhyAddr(offset3), out_three_real);
                WriteState(state_gm.GetPhyAddr(offset3 + 1U), out_three_imag);
                FlushStateCacheLine(state_gm, offset0);
                if ((offset0 & ~15ULL) != (offset1 & ~15ULL)) {
                    FlushStateCacheLine(state_gm, offset1);
                }
                if ((offset0 & ~15ULL) != (offset2 & ~15ULL) && (offset1 & ~15ULL) != (offset2 & ~15ULL)) {
                    FlushStateCacheLine(state_gm, offset2);
                }
                if ((offset0 & ~15ULL) != (offset3 & ~15ULL) && (offset1 & ~15ULL) != (offset3 & ~15ULL)
                    && (offset2 & ~15ULL) != (offset3 & ~15ULL)) {
                    FlushStateCacheLine(state_gm, offset3);
                }
            }
        }
    }
}

extern "C" __global__ __aicore__ void mqvector_apply_local_circuit_cached(
    GM_ADDR state, GM_ADDR gates, GM_ADDR local_indices, uint64_t dim, uint32_t gate_count, uint64_t local_mask,
    uint32_t local_qubits) {
    GlobalTensor<float> state_gm;
    state_gm.SetGlobalBuffer((__gm__ float*)state, dim * 2U);
    __gm__ const CircuitGate* gates_gm = reinterpret_cast<__gm__ const CircuitGate*>(gates);
    __gm__ const uint32_t* local_indices_gm = reinterpret_cast<__gm__ const uint32_t*>(local_indices);

    const uint64_t physical_block_idx = static_cast<uint64_t>(GetBlockIdx());
    if ((physical_block_idx & 1U) != 0U) {
        return;
    }

    const uint64_t block_idx = physical_block_idx >> 1U;
    const uint64_t block_num = (static_cast<uint64_t>(GetBlockNum()) + 1U) >> 1U;
    const uint64_t local_dim = static_cast<uint64_t>(1U) << local_qubits;
    if (local_qubits == 0U || local_qubits > 14U || local_dim > dim) {
        return;
    }

    TPipe pipe;
    TBuf<QuePosition::VECCALC> state_buffer;
    TBuf<QuePosition::VECCALC> index_buffer;
    pipe.InitBuffer(state_buffer, static_cast<uint32_t>(local_dim * 2U * sizeof(float)));
    LocalTensor<float> cached_state_tensor = state_buffer.Get<float>();
    __ubuf__ float* cached_state = (__ubuf__ float*)cached_state_tensor.GetPhyAddr();
    pipe.InitBuffer(index_buffer, static_cast<uint32_t>(local_dim * sizeof(uint32_t)));
    LocalTensor<uint32_t> cached_index_tensor = index_buffer.Get<uint32_t>();
    __ubuf__ uint32_t* cached_indices = (__ubuf__ uint32_t*)cached_index_tensor.GetPhyAddr();
    for (uint64_t local_index = 0; local_index < local_dim; ++local_index) {
        cached_indices[local_index] = local_indices_gm[local_index];
    }

    const uint64_t tile_count = dim >> local_qubits;
    const uint64_t all_state_mask = dim - 1U;
    const uint64_t nonlocal_mask = all_state_mask & ~local_mask;

    for (uint64_t tile_index = block_idx; tile_index < tile_count; tile_index += block_num) {
        const uint64_t tile_prefix = ScatterBits(tile_index, nonlocal_mask);
        for (uint64_t local_index = 0; local_index < local_dim; ++local_index) {
            const uint64_t local_basis = static_cast<uint64_t>(cached_indices[local_index]);
            const uint64_t basis = tile_prefix | local_basis;
            const uint64_t state_offset = basis << 1U;
            const uint64_t cached_offset = local_index << 1U;
            cached_state[cached_offset] = ReadState(state_gm.GetPhyAddr(state_offset));
            cached_state[cached_offset + 1U] = ReadState(state_gm.GetPhyAddr(state_offset + 1U));
        }

        for (uint32_t gate_index = 0; gate_index < gate_count; ++gate_index) {
            __gm__ const CircuitGate* gate = gates_gm + gate_index;
            const uint64_t nonlocal_control_mask = gate->control_mask & nonlocal_mask;
            if ((tile_prefix & nonlocal_control_mask) != nonlocal_control_mask) {
                continue;
            }
            if (gate->kind == 0U) {
                const uint64_t pair_count = local_dim >> 1U;
                const uint64_t target_local_bit = static_cast<uint64_t>(DecodeLocalMeta(gate->cosine));
                const uint64_t lower_local_mask = target_local_bit - 1U;
                const uint64_t local_control_bits = static_cast<uint64_t>(DecodeLocalMeta(gate->sine));
                for (uint64_t pair_index = 0; pair_index < pair_count; ++pair_index) {
                    const uint64_t local_index =
                        (pair_index & lower_local_mask)
                        | ((pair_index & ~lower_local_mask) << 1U);
                    if (local_control_bits != 0U && (local_index & local_control_bits) != local_control_bits) {
                        continue;
                    }

                    const uint64_t zero_offset = local_index << 1U;
                    const uint64_t one_offset = (local_index | target_local_bit) << 1U;
                    ApplyMatrixPairLocal(cached_state, gate, zero_offset, one_offset);
                }
                continue;
            }

            const uint64_t quartet_count = local_dim >> 2U;
            const uint64_t low_local_bit = static_cast<uint64_t>(DecodeLocalMeta(gate->m00_real));
            const uint64_t high_local_bit = static_cast<uint64_t>(DecodeLocalMeta(gate->m00_imag));
            const uint64_t low_local_mask = static_cast<uint64_t>(DecodeLocalMeta(gate->m01_real));
            const uint64_t middle_local_mask = static_cast<uint64_t>(DecodeLocalMeta(gate->m01_imag));
            const uint64_t low_local_pos = static_cast<uint64_t>(DecodeLocalMeta(gate->m10_real));
            const uint64_t high_local_pos = static_cast<uint64_t>(DecodeLocalMeta(gate->m10_imag));
            const uint64_t local_control_bits = static_cast<uint64_t>(DecodeLocalMeta(gate->m11_real));

            for (uint64_t quartet_index = 0; quartet_index < quartet_count; ++quartet_index) {
                const uint64_t low_bits = quartet_index & low_local_mask;
                const uint64_t middle_bits = (quartet_index >> low_local_pos) & middle_local_mask;
                const uint64_t high_bits = quartet_index >> (high_local_pos - 1U);
                const uint64_t local_index =
                    low_bits | (middle_bits << (low_local_pos + 1U)) | (high_bits << (high_local_pos + 1U));
                if (local_control_bits != 0U && (local_index & local_control_bits) != local_control_bits) {
                    continue;
                }

                const uint64_t offset0 = local_index << 1U;
                const uint64_t offset1 = (local_index | low_local_bit) << 1U;
                const uint64_t offset2 = (local_index | high_local_bit) << 1U;
                const uint64_t offset3 = (local_index | low_local_bit | high_local_bit) << 1U;

                if (gate->operation == 0U) {
                    const float one_real = ReadLocal(cached_state, offset1);
                    const float one_imag = ReadLocal(cached_state, offset1 + 1U);
                    const float two_real = ReadLocal(cached_state, offset2);
                    const float two_imag = ReadLocal(cached_state, offset2 + 1U);
                    WriteLocal(cached_state, offset1, two_real);
                    WriteLocal(cached_state, offset1 + 1U, two_imag);
                    WriteLocal(cached_state, offset2, one_real);
                    WriteLocal(cached_state, offset2 + 1U, one_imag);
                    continue;
                }

                const float zero_real = ReadLocal(cached_state, offset0);
                const float zero_imag = ReadLocal(cached_state, offset0 + 1U);
                const float one_real = ReadLocal(cached_state, offset1);
                const float one_imag = ReadLocal(cached_state, offset1 + 1U);
                const float two_real = ReadLocal(cached_state, offset2);
                const float two_imag = ReadLocal(cached_state, offset2 + 1U);
                const float three_real = ReadLocal(cached_state, offset3);
                const float three_imag = ReadLocal(cached_state, offset3 + 1U);

                float out_zero_real;
                float out_zero_imag;
                float out_one_real;
                float out_one_imag;
                float out_two_real;
                float out_two_imag;
                float out_three_real;
                float out_three_imag;

                if (gate->operation == 1U) {
                    out_zero_real = gate->cosine * zero_real + gate->sine * three_imag;
                    out_zero_imag = gate->cosine * zero_imag - gate->sine * three_real;
                    out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                    out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                    out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                    out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                    out_three_real = gate->cosine * three_real + gate->sine * zero_imag;
                    out_three_imag = gate->cosine * three_imag - gate->sine * zero_real;
                } else if (gate->operation == 2U) {
                    out_zero_real = gate->cosine * zero_real - gate->sine * three_imag;
                    out_zero_imag = gate->cosine * zero_imag + gate->sine * three_real;
                    out_one_real = gate->cosine * one_real + gate->sine * two_imag;
                    out_one_imag = gate->cosine * one_imag - gate->sine * two_real;
                    out_two_real = gate->cosine * two_real + gate->sine * one_imag;
                    out_two_imag = gate->cosine * two_imag - gate->sine * one_real;
                    out_three_real = gate->cosine * three_real - gate->sine * zero_imag;
                    out_three_imag = gate->cosine * three_imag + gate->sine * zero_real;
                } else {
                    out_zero_real = gate->cosine * zero_real + gate->sine * zero_imag;
                    out_zero_imag = gate->cosine * zero_imag - gate->sine * zero_real;
                    out_one_real = gate->cosine * one_real - gate->sine * one_imag;
                    out_one_imag = gate->cosine * one_imag + gate->sine * one_real;
                    out_two_real = gate->cosine * two_real - gate->sine * two_imag;
                    out_two_imag = gate->cosine * two_imag + gate->sine * two_real;
                    out_three_real = gate->cosine * three_real + gate->sine * three_imag;
                    out_three_imag = gate->cosine * three_imag - gate->sine * three_real;
                }

                WriteLocal(cached_state, offset0, out_zero_real);
                WriteLocal(cached_state, offset0 + 1U, out_zero_imag);
                WriteLocal(cached_state, offset1, out_one_real);
                WriteLocal(cached_state, offset1 + 1U, out_one_imag);
                WriteLocal(cached_state, offset2, out_two_real);
                WriteLocal(cached_state, offset2 + 1U, out_two_imag);
                WriteLocal(cached_state, offset3, out_three_real);
                WriteLocal(cached_state, offset3 + 1U, out_three_imag);
            }
        }

        for (uint64_t local_index = 0; local_index < local_dim; ++local_index) {
            const uint64_t local_basis = static_cast<uint64_t>(cached_indices[local_index]);
            const uint64_t basis = tile_prefix | local_basis;
            const uint64_t state_offset = basis << 1U;
            const uint64_t cached_offset = local_index << 1U;
            WriteState(state_gm.GetPhyAddr(state_offset), cached_state[cached_offset]);
            WriteState(state_gm.GetPhyAddr(state_offset + 1U), cached_state[cached_offset + 1U]);
        }
    }
}
